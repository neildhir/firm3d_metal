"""Shared pieces of the benchmark plots: runs, machines, legend, and style.

Every kind of benchmark is plotted with the same encoding: colour is the
backend (CPU blue, GPU orange) and line style the machine, both spelled out
in a legend below the figure. Each machine is drawn from its latest run that
passed the sanity check, and keeps its line style for good: styles go to
machines in order of their first run. Runs of different workloads are never
mixed.
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.typing import LineStyleType

import machine as mach

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
            "choose a further encoding in plot_common.py"
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
                "CPU, float64, MPI over particles",
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
