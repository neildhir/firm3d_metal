"""Benchmark plots across machines: CPU (MPI) vs Metal GPU tracing.

Reads every run filed by run_benchmark.py under results/<benchmark>/ and
writes, per benchmark, to plots/<benchmark>/:

- scaling.png: wall time vs ensemble size, CPU and GPU, with a speedup panel
- validation.png: loss-fraction curves, CPU float64 vs Metal float32
- gpu_machines.png: Metal GPU wall time vs ensemble size, machine by machine

Colour is the backend and line style the machine, both spelled out in a
legend below each figure. Each machine is drawn from its latest run that
passed the sanity check, and keeps its line style for good: styles go to
machines in order of their first run. Runs of different workloads (config or
equilibrium fingerprints differ) are never mixed; that is an error.

Each plotted machine also gets a marker-by-marker physics check
(physics_check.py): its CPU against its GPU, and its traces against the
first machine's. A machine that fails is still plotted, but flagged in the
legend and warned about. Prints two Markdown tables per benchmark for the
README: headline numbers and the physics check.

Usage
-----
python make_plots.py [benchmark ...]    (default: every benchmark in results/)
"""

import json
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.typing import LineStyleType

import bench_common as bc
import machine as mach
import physics_check as phys

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
PLOTS_DIR = BENCH_DIR / "plots"
ARCHIVE = "archive"  # results/archive/ holds pre-fingerprint runs; never plotted

# dataviz reference palette (validated): slot 1 blue = CPU, slot 2 orange = GPU
C_CPU = "#2a78d6"
C_GPU = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e5e4e0"

# One line style per machine, in order of the machines' first runs. A fifth
# machine is an error until a further encoding is chosen.
LINESTYLES: tuple[LineStyleType, ...] = ("-", "--", ":", "-.")
ALPHA = 0.7  # lets coincident curves from several machines show through
LINEWIDTH = 2.0
LABEL_GAP = 0.08  # minimum vertical spacing of speedup labels, fraction of the axis
LEGEND_HEADER = Line2D([], [], linestyle="none")


@dataclass(frozen=True)
class Run:
    """One filed benchmark run: its record and both legs' results."""

    folder: Path
    record: dict[str, Any]
    cpu: dict[str, Any]
    metal: dict[str, Any]

    @property
    def created(self) -> datetime:
        return datetime.fromisoformat(self.record["created"])

    @property
    def machine_key(self) -> tuple[str | int, ...]:
        """Contributor plus hardware: two people's identical Macs stay apart."""
        return (self.record["tag"], *mach.hardware_key(self.record["machine"]))

    def final_states(self, leg: str) -> np.ndarray:
        """(nparticles, 5) validation final states of leg "cpu" or "metal"."""
        path = bc.final_states_path(self.folder / f"{leg}.json")
        return np.load(path)["final_states"]


@dataclass(frozen=True)
class Series:
    """A machine as plotted: its latest run, legend label, and line style."""

    run: Run
    label: str
    linestyle: LineStyleType
    physics_failed: bool = False

    @property
    def legend_label(self) -> str:
        """The label, with a second line flagging a failed physics check."""
        if self.physics_failed:
            return f"{self.label}\nphysics check FAILED: see the physics table"
        return self.label


def main(names: list[str]) -> None:
    """Plot each requested benchmark (default: all) and print its table."""
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
        checks = physics_checks(series, cfg)
        series = flag_failures(series, checks)
        outdir = PLOTS_DIR / bench_dir.name
        outdir.mkdir(parents=True, exist_ok=True)
        scaling_plot(series, cfg, outdir)
        validation_plot(series, cfg, outdir)
        gpu_machines_plot(series, cfg, outdir)
        print(f"{bench_dir.name}: {len(series)} machine(s) -> {outdir}\n")
        print(f"{summary_table(series)}\n")
        print(f"{phys.table([s.label for s in series], checks)}\n")


def load_runs(bench_dir: Path) -> list[Run]:
    """All completed runs of one benchmark, oldest first."""
    runs = []
    for folder in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
        if folder.name.endswith(".incomplete") or not (folder / "run.json").exists():
            warn(f"skipping {folder.name}: run did not complete")
            continue
        runs.append(
            Run(
                folder,
                json.loads((folder / "run.json").read_text()),
                json.loads((folder / "cpu.json").read_text()),
                json.loads((folder / "metal.json").read_text()),
            )
        )
    return sorted(runs, key=lambda r: r.created)


def check_comparable(runs: list[Run]) -> None:
    """Refuse to mix workloads; warn when the Metal code differs.

    Raises
    ------
    ValueError
        If the runs' config or equilibrium fingerprints differ.
    """
    for key in ("config", "equilibrium"):
        prints = {r.record["fingerprints"][key] for r in runs}
        if len(prints) > 1:
            listing = "\n".join(
                f"  {r.folder.name}: {r.record['fingerprints'][key][:12]}" for r in runs
            )
            raise ValueError(
                f"runs in {runs[0].folder.parent.name} trace different workloads "
                f"({key} fingerprints differ); give the changed workload its own "
                f"benchmark name:\n{listing}"
            )


def select_series(runs: list[Run]) -> list[Series]:
    """Latest passing run per machine, with stable line styles and labels.

    Raises
    ------
    ValueError
        If there are more machines than line styles.
    """
    passing = []
    for run in runs:
        if run.record["sanity"]["passed"]:
            passing.append(run)
        else:
            warn(f"skipping {run.folder.name}: failed its sanity check")

    by_machine: dict[tuple[str | int, ...], list[Run]] = {}
    for run in passing:  # oldest first, so machines appear by first run
        by_machine.setdefault(run.machine_key, []).append(run)
    if len(by_machine) > len(LINESTYLES):
        raise ValueError(
            f"{len(by_machine)} machines but only {len(LINESTYLES)} line styles "
            f"({' '.join(map(str, LINESTYLES))}); "
            "choose a further encoding in make_plots.py"
        )

    latest = [machine_runs[-1] for machine_runs in by_machine.values()]
    # identical hardware from two contributors is told apart by their tags
    hardware = [mach.hardware_key(r.record["machine"]) for r in latest]
    labels = [
        mach.label(r.record["machine"])
        + (f" [{r.record['tag']}]" if hardware.count(hw) > 1 else "")
        for r, hw in zip(latest, hardware, strict=True)
    ]
    if len({r.record["fingerprints"]["metal_sources"] for r in latest}) > 1:
        warn("the plotted runs used different Metal kernel sources")
    return [
        Series(r, label, style)
        for r, label, style in zip(latest, labels, LINESTYLES, strict=False)
    ]


def physics_checks(
    series: list[Series], cfg: dict[str, Any]
) -> list[phys.MachineCheck]:
    """Marker-by-marker physics check of each machine; the first is reference."""
    traces = [(s.run.final_states("cpu"), s.run.final_states("metal")) for s in series]
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


def add_legend(
    fig: Figure,
    series: list[Series],
    backends: bool = True,
    backend_markers: bool = False,
) -> None:
    """Legend below the axes: backend colours, then machine line styles.

    Grouping the two encodings keeps the legend at (machines + 2) entries
    instead of one entry per machine and backend.
    """
    entries: list[tuple[Line2D, str]] = []
    if backends:
        entries += [
            (LEGEND_HEADER, "Backend"),
            (
                Line2D(
                    [],
                    [],
                    color=C_CPU,
                    linewidth=LINEWIDTH,
                    marker="o" if backend_markers else "",
                ),
                "CPU, float64, MPI with one rank per physical core",
            ),
            (
                Line2D(
                    [],
                    [],
                    color=C_GPU,
                    linewidth=LINEWIDTH,
                    marker="s" if backend_markers else "",
                ),
                "Metal GPU, float32",
            ),
        ]
    entries.append((LEGEND_HEADER, "Machine"))
    entries += [
        (
            Line2D([], [], color=TEXT_2, linewidth=LINEWIDTH, linestyle=s.linestyle),
            s.legend_label,
        )
        for s in series
    ]
    handles, labels = zip(*entries, strict=True)
    legend = fig.legend(
        handles,
        labels,
        loc="outside lower left",
        frameon=False,
        fontsize=9,
        handlelength=3.5,
        labelcolor=TEXT,
    )
    for text, handle in zip(legend.get_texts(), handles, strict=True):
        if handle is LEGEND_HEADER:
            text.set_fontweight("bold")


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


def style_axis(ax: Axes) -> None:
    """Recessive grid and spines, secondary ink for labels."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)


def warn(message: str) -> None:
    """Print a warning to stderr."""
    print(f"warning: {message}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
