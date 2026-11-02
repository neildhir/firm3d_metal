"""Plots and tables of conservation benchmarks (kind "conservation").

Writes to plots/<benchmark>/:

- drift_vs_tolerance.png: median p_eta drift against solver tolerance, one
  panel per field resolution. Where the curves flatten, the field's
  representation (or float32 arithmetic) limits accuracy, not the solver.
- work_precision.png: median drift against tracing wall time for every
  (resolution, tolerance) cell, with each backend's front: the best
  accuracy reached for a given time.

Returns the tables for conservation-qh.md (each machine's best accuracy per
backend and the speedup at the GPU's best accuracy; how each machine's
per-marker drifts compare with the first, reference, machine's) and each
machine's at-a-glance columns for README.md.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

import bench_conservation_common as cc
from plot_common import (
    ALPHA,
    C_CPU,
    C_GPU,
    LINEWIDTH,
    SURFACE,
    TEXT,
    Run,
    Series,
    Tables,
    add_legend,
    headline,
    style_axis,
)

BACKENDS = (("cpu", C_CPU, "o"), ("metal", C_GPU, "s"))  # leg, colour, marker
CELL_ALPHA = 0.3  # the individual cells behind each front in work_precision.png


@dataclass(frozen=True)
class AtGpuBest:
    """One machine's backends compared at the best accuracy the GPU reaches.

    Each backend's cell is its fastest one whose median drift is at most
    the GPU's best, so both deliver the accuracy float32 can reach. The
    speedup with builds adds each cell's field build (GPU table, CPU
    interpolant): the worst case, a build per trace, where in production
    one build serves many traces.
    """

    best_cpu: dict[str, Any]  # the CPU's most accurate cell
    best_gpu: dict[str, Any]  # the GPU's most accurate cell
    cpu: dict[str, Any] | None  # fastest CPU cell at the GPU's best, if any
    gpu: dict[str, Any]  # fastest GPU cell at its own best

    @property
    def speedup(self) -> float | None:
        """Tracing time, CPU / GPU."""
        if self.cpu is None:
            return None
        return self.cpu["wall_s"] / self.gpu["wall_s"]

    @property
    def speedup_with_builds(self) -> float | None:
        """Field build plus tracing time, CPU / GPU."""
        if self.cpu is None:
            return None
        total = self.cpu["field_build_s"] + self.cpu["wall_s"]
        return total / (self.gpu["field_build_s"] + self.gpu["wall_s"])


def plot(series: list[Series], cfg: dict[str, Any], outdir: Path) -> Tables:
    """Both figures, the page's tables, and the at-a-glance columns."""
    drift_vs_tolerance_plot(series, cfg, outdir)
    work_precision_plot(series, cfg, outdir)
    return Tables(
        page={"summary": summary_table(series), "agreement": agreement_table(series)},
        headline=dict(headline(s, glance_columns(at_gpu_best(s.run))) for s in series),
    )


def at_gpu_best(run: Run) -> AtGpuBest:
    """Compare a run's backends at the GPU's best accuracy."""
    cpu, gpu = leg_cells(run, "cpu"), leg_cells(run, "metal")
    best_gpu = best_cell(gpu)
    target = best_gpu["drift"]["p50"]
    fast_gpu = fastest_reaching(gpu, target)
    if fast_gpu is None:  # cannot happen: the best GPU cell reaches the target
        raise RuntimeError(f"{run.folder.name}: no GPU cell reaches its own best")
    return AtGpuBest(best_cell(cpu), best_gpu, fastest_reaching(cpu, target), fast_gpu)


def glance_columns(cmp: AtGpuBest) -> dict[str, str]:
    """A machine's conservation columns of the at-a-glance table."""
    if cmp.speedup is None or cmp.speedup_with_builds is None:
        speedup = "—"
    else:
        speedup = f"{cmp.speedup:.1f}× ({cmp.speedup_with_builds:.1f}×)"
    return {
        "Conservation: best median drift, CPU / GPU": (
            f"{cmp.best_cpu['drift']['p50']:.1e} / {cmp.best_gpu['drift']['p50']:.1e}"
        ),
        "Conservation: speedup at the GPU's best (with builds)": speedup,
    }


def drift_vs_tolerance_plot(
    series: list[Series], cfg: dict[str, Any], outdir: Path
) -> None:
    """Median drift against tolerance, one panel per resolution."""
    resolutions = [int(r) for r in cfg["resolutions"]]
    fig, axes = plt.subplots(
        1,
        len(resolutions),
        figsize=(9.6, 5.6),
        sharey=True,
        layout="constrained",
        facecolor=SURFACE,
    )
    for ax, res in zip(axes, resolutions, strict=True):
        for s in series:
            for leg, color, marker in BACKENDS:
                cells = [c for c in leg_cells(s.run, leg) if c["resolution"] == res]
                ax.loglog(
                    [c["tol"] for c in cells],
                    [c["drift"]["p50"] for c in cells],
                    marker=marker,
                    color=color,
                    linestyle=s.linestyle,
                    linewidth=LINEWIDTH,
                    alpha=ALPHA,
                )
        ax.invert_xaxis()  # tighter tolerances to the right
        ax.set_title(f"{res}³ field grid", color=TEXT, fontsize=10)
        ax.set_xlabel("solver tolerance", color=TEXT, fontsize=10)
        style_axis(ax)
    axes[0].set_ylabel(
        "median p$_\\eta$ drift / (q ψ$_{edge}$)", color=TEXT, fontsize=10
    )
    fig.suptitle(
        "p$_\\eta$ drift against solver tolerance\n" + workload(cfg),
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=11,
    )
    add_legend(fig, series, backend_markers=True)
    fig.savefig(outdir / "drift_vs_tolerance.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def work_precision_plot(
    series: list[Series], cfg: dict[str, Any], outdir: Path
) -> None:
    """Median drift against tracing time: all cells, and each backend's front."""
    fig, ax = plt.subplots(figsize=(7.2, 6.0), layout="constrained", facecolor=SURFACE)
    for s in series:
        for leg, color, marker in BACKENDS:
            cells = leg_cells(s.run, leg)
            times = np.array([c["wall_s"] for c in cells])
            drifts = np.array([c["drift"]["p50"] for c in cells])
            ax.loglog(
                times, drifts, marker, color=color, alpha=CELL_ALPHA, markersize=5
            )
            front_t, front_d = front(times, drifts)
            ax.loglog(
                front_t,
                front_d,
                marker=marker,
                color=color,
                linestyle=s.linestyle,
                linewidth=LINEWIDTH,
                alpha=ALPHA,
            )
    ax.set_xlabel("tracing wall time [s]", color=TEXT, fontsize=10)
    ax.set_ylabel("median p$_\\eta$ drift / (q ψ$_{edge}$)", color=TEXT, fontsize=10)
    ax.set_title(
        "Accuracy against time (faint: every resolution × tolerance)\n" + workload(cfg),
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    style_axis(ax)
    add_legend(fig, series, backend_markers=True)
    fig.savefig(outdir / "work_precision.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def front(times: np.ndarray, drifts: np.ndarray) -> tuple[list[float], list[float]]:
    """The cells no other cell beats on both time and accuracy, fastest first."""
    xs: list[float] = []
    ys: list[float] = []
    for i in np.argsort(times):
        if not ys or drifts[i] < ys[-1]:
            xs.append(float(times[i]))
            ys.append(float(drifts[i]))
    return xs, ys


def summary_table(series: list[Series]) -> str:
    """Markdown table: best accuracy per backend, speedup at the GPU's best.

    See AtGpuBest. Wrapped in markdownlint comments lifting the line-length
    rule, like the other tables.
    """
    rows = [
        "<!-- markdownlint-disable MD013 -->",
        "| Machine | CPU ranks | Best median drift, CPU | Best median drift, GPU "
        "| Tracing to the GPU's best, CPU / GPU | Speedup | With field builds |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in series:
        cmp = at_gpu_best(s.run)
        if cmp.cpu is None or cmp.speedup is None or cmp.speedup_with_builds is None:
            timing = "— | — | —"
        else:
            timing = (
                f"{cmp.cpu['wall_s']:.2f} s ({cell_name(cmp.cpu)}) / "
                f"{cmp.gpu['wall_s']:.2f} s ({cell_name(cmp.gpu)}) "
                f"| {cmp.speedup:.1f}× | {cmp.speedup_with_builds:.1f}×"
            )
        rows.append(
            f"| {s.label} | {s.run.record['mpi_ranks']} "
            f"| {cmp.best_cpu['drift']['p50']:.1e} ({cell_name(cmp.best_cpu)}) "
            f"| {cmp.best_gpu['drift']['p50']:.1e} ({cell_name(cmp.best_gpu)}) "
            f"| {timing} |"
        )
    rows.append("<!-- markdownlint-enable MD013 -->")
    return "\n".join(rows)


def agreement_table(series: list[Series]) -> str:
    """Markdown table: each machine's per-marker drifts against the reference's.

    Compares every marker in every cell: "identical" when the drifts match
    bit for bit, otherwise the median ratio (this machine / reference) and
    the Wilcoxon signed-rank p for a systematic difference.
    """
    rows = [
        "<!-- markdownlint-disable MD013 -->",
        "| Machine | GPU drifts vs reference GPU | CPU drifts vs reference CPU |",
        "| --- | --- | --- |",
    ]
    ref = series[0].run
    for i, s in enumerate(series):
        if i == 0:
            cells = ["reference", "reference"]
        else:
            cells = [
                compare_drifts(drifts(ref, leg), drifts(s.run, leg))
                for leg in ("metal", "cpu")
            ]
        rows.append(f"| {s.label} | {' | '.join(cells)} |")
    rows.append("<!-- markdownlint-enable MD013 -->")
    return "\n".join(rows)


def compare_drifts(a: np.ndarray, b: np.ndarray) -> str:
    """One agreement-table cell: identical, or the median ratio b / a and p."""
    if np.array_equal(a, b):
        return "identical"
    both = (a > 0) & (b > 0)
    log_ratio = np.log(b[both] / a[both])
    shifted = log_ratio[log_ratio != 0.0]
    p = 1.0
    if shifted.size:
        result = wilcoxon(shifted)
        # scipy's stubs do not declare the result's pvalue attribute
        p = float(result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]
    return f"median ratio ×{np.exp(np.median(log_ratio)):.3f}, p {p:.2g}"


def leg_cells(run: Run, leg: str) -> list[dict[str, Any]]:
    """The (resolution, tolerance) cells of leg "cpu" or "metal"."""
    return (run.cpu if leg == "cpu" else run.metal)["cells"]


def drifts(run: Run, leg: str) -> np.ndarray:
    """(nresolutions, ntolerances, nparticles) per-marker drifts of a leg."""
    return np.load(cc.drift_path(run.folder / f"{leg}.json"))["drift"]


def best_cell(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """The cell with the lowest median drift."""
    return min(cells, key=lambda c: c["drift"]["p50"])


def fastest_reaching(
    cells: list[dict[str, Any]], target: float
) -> dict[str, Any] | None:
    """The fastest cell whose median drift is at most target, if any."""
    reaching = [c for c in cells if c["drift"]["p50"] <= target]
    return min(reaching, key=lambda c: c["wall_s"]) if reaching else None


def cell_name(cell: dict[str, Any]) -> str:
    """Short name of a cell, e.g. "64³, tol 1e-8"."""
    tol = f"{cell['tol']:g}".replace("e-0", "e-")
    return f"{cell['resolution']}³, tol {tol}"


def workload(cfg: dict[str, Any]) -> str:
    """Two-line description of the workload, for plot titles."""
    tracing = cfg["tracing"]
    return (
        f"{cfg['description']}\n"
        f"{int(cfg['nparticles']):,} markers · tmax = {tracing['tmax_s'] * 1e3:g} ms "
        f"· p$_\\eta$ checked every {tracing['dt_save_s'] * 1e6:g} µs"
    )
