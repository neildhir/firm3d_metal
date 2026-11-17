"""Drive the layout experiment: a bitwise smoke test, then timed rounds.

    python run_layout.py config.yaml smoke   # short traces; bitwise agreement
    python run_layout.py config.yaml timed   # appends to <data_dir>/results.csv
Needs bin/layout_bench (built from layout_bench.mm) and the inputs from prep_inputs.py.
"""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from layout_config import LayoutConfig, load_config  # noqa: E402

HERE = Path(__file__).parent
HARNESS = HERE / "bin" / "layout_bench"
VARIANTS_SRC = HERE / "variants.metal"
FIELDS = [
    "variant",
    "n",
    "tg",
    "steps_per_dispatch",
    "tol",
    "tmax_s",
    "gpu_s",
    "wall_s",
    "dispatches",
    "longest_dispatch_s",
    "total_steps",
    "lost",
    "round",
]


def run_once(
    cfg: LayoutConfig, variant: str, n: int, tmax: float, tg: int, out_prefix: Path
) -> dict[str, str]:
    """Run the harness once; its CSV row as a dict, without the round column."""
    cmd = [
        str(HARNESS),
        str(cfg.kernel_source),
        str(VARIANTS_SRC),
        str(cfg.data_dir),
        variant,
        str(n),
        repr(tmax),
        repr(cfg.tol),
        str(cfg.steps_per_dispatch),
        str(tg),
        str(out_prefix),
    ]
    out = (
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        .stdout.strip()
        .splitlines()[-1]
    )
    return dict(zip(FIELDS[:-1], out.split(","), strict=True))


def configurations(cfg: LayoutConfig) -> list[tuple[str, int]]:
    """(variant, threadgroup size) pairs; the one-thread kernel runs at each size."""
    pairs = []
    for v in cfg.variants:
        pairs.extend((v, tg) for tg in (cfg.threads_per_tg_1t if v == "1t" else (32,)))
    return pairs


def smoke(cfg: LayoutConfig) -> None:
    """Short traces with every variant; do final states and steps match bitwise?"""
    outdir = cfg.data_dir / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    finals: dict[str, np.ndarray] = {}
    steps: dict[str, np.ndarray] = {}
    for variant, tg in configurations(cfg):
        key = f"{variant}-tg{tg}"
        row = run_once(
            cfg, variant, cfg.smoke_nparticles, cfg.smoke_tmax_s, tg, outdir / key
        )
        finals[key] = np.fromfile(
            outdir / f"{key}.final.f32", dtype=np.float32
        ).reshape(-1, 6)
        steps[key] = np.fromfile(outdir / f"{key}.steps.u32", dtype=np.uint32)
        print(
            f"{key}: steps {row['total_steps']}, lost {row['lost']}, "
            f"gpu {float(row['gpu_s']):.3f} s"
        )
    ref = "orig-tg32"
    for key, fin in finals.items():
        same = np.array_equal(fin.view(np.uint32), finals[ref].view(np.uint32))
        ndiff = int(
            np.sum(np.any(fin.view(np.uint32) != finals[ref].view(np.uint32), axis=1))
        )
        print(
            f"{key} vs {ref}: final states bitwise identical {same} "
            f"({ndiff} particles differ)"
        )
    for key in steps:
        if key != ref:
            print(
                f"{key} vs coop-tg32: step counts identical "
                f"{np.array_equal(steps[key], steps['coop-tg32'])}"
            )


def timed(cfg: LayoutConfig) -> None:
    """Timed rounds: every (n, variant) once per round, in turn; appends rows."""
    results = cfg.data_dir / "results.csv"
    new = not results.exists()
    outdir = cfg.data_dir / "timed"
    outdir.mkdir(parents=True, exist_ok=True)
    with results.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for rnd in range(cfg.repeats):
            for n in cfg.nparticles:
                for variant, tg in configurations(cfg):
                    row = run_once(
                        cfg,
                        variant,
                        n,
                        cfg.tmax_s,
                        tg,
                        outdir / f"{variant}-tg{tg}-n{n}-r{rnd}",
                    )
                    row["round"] = str(rnd)
                    writer.writerow(row)
                    f.flush()
                    print(",".join(row[k] for k in FIELDS), flush=True)


if __name__ == "__main__":
    config = load_config(Path(sys.argv[1]))
    {"smoke": smoke, "timed": timed}[sys.argv[2]](config)
