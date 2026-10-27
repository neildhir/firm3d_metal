"""Benchmark comparison plots: CPU (MPI) vs Metal GPU tracing on this Mac.

Reads the two benchmark JSON files and produces:

- scaling.png: wall time vs ensemble size (log-log) with a speedup panel
- validation.png: loss-fraction curves, CPU float64 vs Metal float32

Usage
-----
python make_plots.py cpu_results.json metal_results.json <outdir>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# dataviz reference palette (validated): slot 1 blue = CPU, slot 2 orange = GPU
C_CPU = "#2a78d6"
C_GPU = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e5e4e0"


def style_axis(ax: plt.Axes) -> None:
    """Recessive grid and spines, secondary ink for labels."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)


def scaling_plot(cpu: dict, gpu: dict, outdir: Path) -> None:
    """Wall time vs ensemble size with a speedup panel below."""
    n = np.array([r["nparticles"] for r in cpu["scaling"]])
    t_cpu = np.array([r["wall_s"] for r in cpu["scaling"]])
    t_gpu = np.array([r["wall_s"] for r in gpu["scaling"]])
    tmax = cpu["scaling"][0]["tmax_s"]

    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.4),
        height_ratios=[2.4, 1.0],
        sharex=True,
        facecolor=SURFACE,
    )

    ax.loglog(n, t_cpu, "-o", color=C_CPU, linewidth=2, markersize=6)
    ax.loglog(n, t_gpu, "-s", color=C_GPU, linewidth=2, markersize=6)
    ax.annotate(
        f"CPU, {cpu['ranks']} MPI ranks (float64)",
        (n[-1], t_cpu[-1]),
        textcoords="offset points",
        xytext=(-14, -4),
        ha="right",
        va="top",
        color=C_CPU,
        fontsize=10,
        fontweight="bold",
    )
    ax.annotate(
        "Metal GPU (float32)",
        (n[-1], t_gpu[-1]),
        textcoords="offset points",
        xytext=(-6, 12),
        ha="right",
        color=C_GPU,
        fontsize=10,
        fontweight="bold",
    )
    ax.set_ylabel("wall time [s]", color=TEXT, fontsize=10)
    ax.set_title(
        f"Alpha-particle tracing on one Apple M4: wall time, "
        f"tmax = {tmax:g} s, tol {cpu['tol']:g}",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    style_axis(ax)

    speedup = t_cpu / t_gpu
    ax2.semilogx(n, speedup, "-o", color=TEXT_2, linewidth=2, markersize=6)
    for x, y in zip(n, speedup):
        ax2.annotate(
            f"{y:.1f}x",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color=TEXT,
            fontsize=9,
        )
    ax2.set_xlabel("Monte Carlo samples", color=TEXT, fontsize=10)
    ax2.set_ylabel("CPU / GPU", color=TEXT, fontsize=10)
    ax2.set_ylim(0, max(speedup) * 1.35)
    style_axis(ax2)

    fig.tight_layout()
    fig.savefig(outdir / "scaling.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def validation_plot(cpu: dict, gpu: dict, outdir: Path) -> None:
    """Loss-fraction curves of the validation ensemble, CPU vs GPU."""
    v_cpu, v_gpu = cpu["validation"], gpu["validation"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    ax.loglog(
        v_cpu["loss_times_s"],
        v_cpu["loss_fraction"],
        "-",
        color=C_CPU,
        linewidth=2.2,
    )
    ax.loglog(
        v_gpu["loss_times_s"],
        v_gpu["loss_fraction"],
        "--",
        color=C_GPU,
        linewidth=2.2,
    )
    # direct labels anchored at different points along the (near-coincident)
    # curves so neither label sits on the other or on the data
    t_c = np.array(v_cpu["loss_times_s"])
    f_c = np.array(v_cpu["loss_fraction"])
    i_mid = int(np.searchsorted(t_c, 1.2e-3))
    ax.annotate(
        f"CPU float64 ({v_cpu['loss_fraction'][-1]:.2%})",
        (t_c[i_mid], f_c[i_mid]),
        textcoords="offset points",
        xytext=(0, 16),
        ha="center",
        color=C_CPU,
        fontsize=10,
        fontweight="bold",
    )
    t_g = np.array(v_gpu["loss_times_s"])
    f_g = np.array(v_gpu["loss_fraction"])
    i_g = int(np.searchsorted(t_g, 3e-3))
    ax.annotate(
        f"Metal float32 ({v_gpu['loss_fraction'][-1]:.2%})",
        (t_g[i_g], f_g[i_g]),
        textcoords="offset points",
        xytext=(0, -40),
        ha="center",
        color=C_GPU,
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("time [s]", color=TEXT, fontsize=10)
    ax.set_ylabel("fraction of lost particles", color=TEXT, fontsize=10)
    ax.set_title(
        f"Loss fraction, {v_cpu['nparticles']} fusion-birth alphas, "
        f"Wistell-A vacuum field (ARIES-CS scale)",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(outdir / "validation.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(cpu_path: Path, gpu_path: Path, outdir: Path) -> None:
    """Build both figures and print the headline numbers."""
    cpu = json.loads(cpu_path.read_text())
    gpu = json.loads(gpu_path.read_text())
    outdir.mkdir(parents=True, exist_ok=True)

    scaling_plot(cpu, gpu, outdir)
    validation_plot(cpu, gpu, outdir)

    v_cpu, v_gpu = cpu["validation"], gpu["validation"]
    print(f"validation workload ({v_cpu['nparticles']} x {v_cpu['tmax_s']:g} s):")
    print(
        f"  CPU : {v_cpu['wall_s'] / 60:.1f} min, loss {v_cpu['loss_fraction'][-1]:.4f}"
    )
    print(
        f"  GPU : {v_gpu['wall_s'] / 60:.1f} min, loss {v_gpu['loss_fraction'][-1]:.4f}"
    )
    print(f"  speedup: {v_cpu['wall_s'] / v_gpu['wall_s']:.1f}x")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
