"""Benchmark plots across machines: CPU (MPI) vs Metal GPU tracing.

Reads every run filed by run_benchmark.py under results/<benchmark>/ and
writes figures per benchmark to plots/<benchmark>/, by kind (kinds.py):

- loss (this module): scaling.png (wall time vs ensemble size, with a
  speedup panel), validation.png (loss-fraction curves), and
  gpu_machines.png (GPU wall time, machine by machine)
- conservation (plots_conservation.py): p_eta drift against tolerance
  and resolution, and accuracy against tracing time

The shared encoding, run selection, and comparability rules are in
plot_common.py. For loss benchmarks each plotted machine also gets a
marker-by-marker physics check (physics_check.py): its CPU against its GPU,
and its traces against the first machine's; a machine that fails is still
plotted, but flagged in the legend and warned about. Prints Markdown tables
per benchmark for the README.

Usage
-----
python make_plots.py [benchmark ...]    (default: every benchmark in results/)
"""

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import bench_common as bc
import kinds
import physics_check as phys
import plots_conservation
from plot_common import (
    ALPHA,
    C_CPU,
    C_GPU,
    LINEWIDTH,
    SURFACE,
    TEXT,
    TEXT_2,
    Run,
    Series,
    add_legend,
    check_comparable,
    load_runs,
    select_series,
    style_axis,
    warn,
)

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
PLOTS_DIR = BENCH_DIR / "plots"
ARCHIVE = "archive"  # results/archive/ holds pre-fingerprint runs; never plotted
LABEL_GAP = 0.08  # minimum vertical spacing of speedup labels, fraction of the axis


def main(names: list[str]) -> None:
    """Plot each requested benchmark (default: all) and print its tables."""
    if names:
        bench_dirs = [RESULTS_DIR / name for name in names]
    else:
        bench_dirs = sorted(
            d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name != ARCHIVE
        )
    for bench_dir in bench_dirs:
        runs = load_runs(bench_dir)
        if not runs:
            warn(f"{bench_dir.name}: no completed runs")
            continue
        check_comparable(runs)
        series = select_series(runs)
        cfg = series[0].run.record["config"]
        outdir = PLOTS_DIR / bench_dir.name
        outdir.mkdir(parents=True, exist_ok=True)
        if kinds.kind_of(cfg) == "conservation":
            tables = plots_conservation.plot(series, cfg, outdir)
        else:
            tables = plot_loss(series, cfg, outdir)
        print(f"{bench_dir.name}: {len(series)} machine(s) -> {outdir}\n")
        for table in tables:
            print(f"{table}\n")


def plot_loss(series: list[Series], cfg: dict[str, Any], outdir: Path) -> list[str]:
    """Loss benchmark: physics check, three figures, and two README tables."""
    checks = physics_checks(series, cfg)
    series = flag_failures(series, checks)
    scaling_plot(series, cfg, outdir)
    validation_plot(series, cfg, outdir)
    gpu_machines_plot(series, cfg, outdir)
    return [summary_table(series), phys.table([s.label for s in series], checks)]


def final_states(run: Run, leg: str) -> np.ndarray:
    """(nparticles, 5) validation final states of leg "cpu" or "metal"."""
    return np.load(bc.final_states_path(run.folder / f"{leg}.json"))["final_states"]


def physics_checks(
    series: list[Series], cfg: dict[str, Any]
) -> list[phys.MachineCheck]:
    """Marker-by-marker physics check of each machine; the first is reference."""
    traces = [
        (final_states(s.run, "cpu"), final_states(s.run, "metal")) for s in series
    ]
    return phys.check_machines(traces, cfg["validation"]["tmax_s"])


def flag_failures(
    series: list[Series], checks: list[phys.MachineCheck]
) -> list[Series]:
    """Mark machines that fail the physics check, for their legend entries."""
    flagged = []
    for s, check in zip(series, checks, strict=True):
        if check.passed:
            flagged.append(s)
            continue
        warn(f"{s.run.folder.name}: physics check FAILED (see the physics table)")
        flagged.append(replace(s, physics_failed=True))
    return flagged


def scaling_plot(series: list[Series], cfg: dict[str, Any], outdir: Path) -> None:
    """Wall time vs ensemble size, CPU and GPU, with a speedup panel below."""
    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.8),
        height_ratios=[2.4, 1.0],
        sharex=True,
        layout="constrained",
        facecolor=SURFACE,
    )
    finals = []  # (n, speedup) at each machine's largest ensemble
    for s in series:
        n, t_cpu, t_gpu = scaling_arrays(s.run)
        line = {"linestyle": s.linestyle, "linewidth": LINEWIDTH, "alpha": ALPHA}
        ax.loglog(n, t_cpu, marker="o", color=C_CPU, **line)
        ax.loglog(n, t_gpu, marker="s", color=C_GPU, **line)
        speedup = t_cpu / t_gpu
        ax2.semilogx(n, speedup, marker="o", color=TEXT_2, **line)
        finals.append((n[-1], speedup[-1]))
    ax.set_ylabel("wall time [s]", color=TEXT, fontsize=10)
    ax.set_title(
        "Wall time vs ensemble size\n" + workload(cfg, cfg["tracing"]["tmax_s"]),
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    ax2.set_xlabel("Monte Carlo samples", color=TEXT, fontsize=10)
    ax2.set_ylabel("speedup, CPU / GPU", color=TEXT, fontsize=10)
    ax2.set_ylim(0, 1.25 * max(y for _, y in finals))
    ax2.margins(x=0.1)
    label_final_speedups(ax2, finals)
    for a in (ax, ax2):
        style_axis(a)
    add_legend(fig, series, backend_markers=True)
    fig.savefig(outdir / "scaling.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def validation_plot(series: list[Series], cfg: dict[str, Any], outdir: Path) -> None:
    """Loss-fraction curves of the validation ensemble, CPU vs GPU."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0), layout="constrained", facecolor=SURFACE)
    for s in series:
        line = {"linestyle": s.linestyle, "linewidth": LINEWIDTH, "alpha": ALPHA}
        for leg, color in ((s.run.cpu, C_CPU), (s.run.metal, C_GPU)):
            v = leg["validation"]
            ax.loglog(v["loss_times_s"], v["loss_fraction"], color=color, **line)
    val = cfg["validation"]
    ax.set_xlabel("time [s]", color=TEXT, fontsize=10)
    ax.set_ylabel("fraction of lost particles", color=TEXT, fontsize=10)
    ax.set_title(
        f"Loss fraction, {val['nparticles']:,} markers\n"
        + workload(cfg, val["tmax_s"]),
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    style_axis(ax)
    add_legend(fig, series, backend_markers=False)
    fig.savefig(outdir / "validation.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def gpu_machines_plot(series: list[Series], cfg: dict[str, Any], outdir: Path) -> None:
    """Metal GPU wall time vs ensemble size, one line per machine."""
    fig, ax = plt.subplots(figsize=(7.2, 5.6), layout="constrained", facecolor=SURFACE)
    for s in series:
        n, _, t_gpu = scaling_arrays(s.run)
        ax.loglog(
            n,
            t_gpu,
            marker="s",
            color=C_GPU,
            linestyle=s.linestyle,
            linewidth=LINEWIDTH,
            alpha=ALPHA,
        )
    ax.set_xlabel("Monte Carlo samples", color=TEXT, fontsize=10)
    ax.set_ylabel("Metal GPU wall time [s]", color=TEXT, fontsize=10)
    ax.set_title(
        "Metal GPU wall time by machine\n" + workload(cfg, cfg["tracing"]["tmax_s"]),
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    style_axis(ax)
    add_legend(fig, series, backends=False)
    fig.savefig(outdir / "gpu_machines.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def summary_table(series: list[Series]) -> str:
    """Markdown table of each machine's headline numbers.

    Wrapped in markdownlint comments that lift the line-length rule: the
    full machine labels are the point of the table and do not fit in 80
    columns.
    """
    first = series[0].run
    big_cpu = first.cpu["scaling"][-1]
    val = first.cpu["validation"]
    big = f"{big_cpu['nparticles']:,} × {big_cpu['tmax_s'] * 1e3:g} ms"
    valid = f"{val['nparticles']:,} × {val['tmax_s'] * 1e3:g} ms"
    rows = [
        "<!-- markdownlint-disable MD013 -->",
        f"| Machine | CPU ranks | {big}: CPU / GPU | Speedup "
        f"| {valid}: CPU / GPU | Speedup | Lost, CPU / GPU |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in series:
        c_big, g_big = s.run.cpu["scaling"][-1], s.run.metal["scaling"][-1]
        c_val, g_val = s.run.cpu["validation"], s.run.metal["validation"]
        rows.append(
            f"| {s.label} | {s.run.record['mpi_ranks']} "
            f"| {c_big['wall_s']:.1f} s / {g_big['wall_s']:.1f} s "
            f"| {c_big['wall_s'] / g_big['wall_s']:.1f}× "
            f"| {c_val['wall_s']:.1f} s / {g_val['wall_s']:.1f} s "
            f"| {c_val['wall_s'] / g_val['wall_s']:.1f}× "
            f"| {c_val['n_lost']} / {g_val['n_lost']} |"
        )
    rows.append("<!-- markdownlint-enable MD013 -->")
    return "\n".join(rows)


def label_final_speedups(ax: Axes, finals: list[tuple[float, float]]) -> None:
    """Label each machine's final speedup, nudging labels apart to never overlap."""
    gap = LABEL_GAP * ax.get_ylim()[1]
    previous = -np.inf
    for x, y in sorted(finals, key=lambda p: p[1]):
        y_text = max(y, previous + gap)
        previous = y_text
        ax.annotate(
            f"{y:.1f}×",
            (x, y),
            xytext=(x * 1.12, y_text),
            ha="left",
            va="center",
            color=TEXT,
            fontsize=9,
        )


def scaling_arrays(run: Run) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensemble sizes and CPU and GPU wall times [s] of a run's scaling leg."""
    n = np.array([r["nparticles"] for r in run.cpu["scaling"]])
    t_cpu = np.array([r["wall_s"] for r in run.cpu["scaling"]])
    t_gpu = np.array([r["wall_s"] for r in run.metal["scaling"]])
    return n, t_cpu, t_gpu


def workload(cfg: dict[str, Any], tmax_s: float) -> str:
    """Two-line description of a workload, for plot titles."""
    res = cfg["interpolation"]["resolution"]
    tol = f"{cfg['tracing']['tol']:g}".replace("e-0", "e-")
    return (
        f"{cfg['description']}\n"
        f"tmax = {tmax_s * 1e3:g} ms · tol {tol} · {res}³ field table"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
