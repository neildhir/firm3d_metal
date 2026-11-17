"""Summarise results.csv: GPU time per DP5 step for each kernel variant.

    python analyze.py config.yaml
orig counts no steps; it takes coop's step count, which smoke tests showed is identical.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent))
from layout_config import load_config  # noqa: E402


def main(results: Path) -> None:
    """Print one row per (n, variant, threadgroup size) with medians over rounds."""
    rows = list(csv.DictReader(results.open()))
    groups: dict[tuple[int, str, int], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[(int(r["n"]), r["variant"], int(r["tg"]))].append(r)

    print(
        f"{'n':>6} {'variant':>12} {'tg':>4} {'rounds':>6} {'gpu_s':>8} "
        f"{'spread':>7} {'wall_s':>8} "
        f"{'Msteps':>8} {'ns/step':>8} {'vs orig':>8} {'lost':>5} {'max_disp_ms':>11}"
    )
    for n in sorted({k[0] for k in groups}):
        coop_steps = median(float(r["total_steps"]) for r in groups[(n, "coop", 32)])
        orig_gpu = median(float(r["gpu_s"]) for r in groups[(n, "orig", 32)])
        for (gn, variant, tg), rs in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])
        ):
            if gn != n:
                continue
            gpus = [float(r["gpu_s"]) for r in rs]
            gpu = median(gpus)
            steps = (
                coop_steps
                if variant == "orig"
                else median(float(r["total_steps"]) for r in rs)
            )
            spread = (max(gpus) - min(gpus)) / gpu
            print(
                f"{n:>6} {variant:>12} {tg:>4} {len(rs):>6} {gpu:>8.3f} {spread:>7.1%} "
                f"{median(float(r['wall_s']) for r in rs):>8.3f} "
                f"{steps / 1e6:>8.3f} {1e9 * gpu / steps:>8.1f} "
                f"{orig_gpu / gpu:>7.2f}x {rs[0]['lost']:>5} "
                f"{1e3 * max(float(r['longest_dispatch_s']) for r in rs):>11.1f}"
            )


if __name__ == "__main__":
    main(load_config(Path(sys.argv[1])).data_dir / "results.csv")
