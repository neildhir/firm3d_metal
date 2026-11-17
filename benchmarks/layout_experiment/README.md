# Metal trace-kernel layout experiment

A controlled comparison, run on an M4 (10 GPU cores) on 17 November 2026, of
three layouts for the Metal trace kernel, all with the same numerics:

- the current cooperative layout (8 particles per 32-lane SIMD-group, as
  CUDA's 32-thread block);
- the same layout with finished particles skipped in the DP5 stages;
- one thread per particle, the layout of sumaddury/firm3d's `metal-port`
  branch.

Skipping finished particles was 1.33-1.41x faster per DP5 step from a few
thousand particles up, with bitwise identical results, and became commit
`9a768a59`. One thread per particle was about 1.3x faster than the old kernel
only up to 1,000 particles, and about 1.03x at 8,000-20,000. The write-up,
with a chart, is the "Layout experiment" section of the review of the two
Metal ports: <https://claude.ai/artifact/1obRGRiYEZPFotD14TYo2A>.

## Contents

- `variants.metal`: kernel variants, compiled after
  `src/firm3dpp/metal_kernel.metal`.
- `layout_bench.mm`: Objective-C++ harness; traces one ensemble with one
  variant and prints one CSV row.
- `prep_inputs.py`: writes the field table and markers as raw files for the
  harness.
- `run_layout.py`: the bitwise smoke test, and timed rounds over all variants.
- `analyze.py`: GPU time per DP5 step for each variant, from `results.csv`.
- `config.yaml`, `config_small.yaml`, `config_fb.yaml`: the vacuum, small-n
  vacuum and finite-beta sweeps.
- `run_all.sh`: all of the above, end to end.
- `results/`: the 17 November results behind the numbers above.
- `head_to_head/`: the comparison with sumaddury's build itself (see below).

## Variants

- `orig`: `trace_boozer` from the checkout's `metal_kernel.metal`.
- `coop`: a copy of `trace_boozer` that also counts steps.
- `coop_active`: `coop` with the DP5 stages gated on still-active particles.
- `1t`: one thread per particle, with the same step controller,
  first-same-as-last reuse, `dtmax` cap, `zeta` wrap and `v_par` tolerance
  scaling, and the same 8-lane reduction order in the interpolation. It is
  not bitwise identical to the others: after one step 999 of 1,000 vacuum
  particles match, and the rest differ by at most 2 ulps, from the compiler
  rounding a few sums differently.

Since `9a768a59` the checkout's kernel skips finished particles, so on a
re-run `orig` behaves like `coop_active`, and the old kernel is `coop`.
`analyze.py` still reports speed-ups against `orig`.

## Running it

```sh
benchmarks/layout_experiment/run_all.sh
```

It needs a checkout installed with `install/macOS/install_firm3d_metal.sh` and
the Xcode command-line tools, and takes about 6 minutes on an M4. It writes
about 0.5 GB of inputs into gitignored `data*/` directories and a
`summary-<host>.txt` beside the script.

The files were moved here from a temporary directory after the runs. Their
paths were made relative at the same time, and `run_all.sh` has not yet been
run end to end from this location.

## Results

`results/vacuum.csv`, `results/vacuum_small.csv` and `results/finite_beta.csv`
hold one row per run: GPU time and wall time in seconds, the number of
dispatches and the longest one in seconds, the total DP5 steps, and the
particles lost. Every run used 128 steps per dispatch, 1 ms traces, tol 1e-6,
and two rounds. `results/chart_rows.json` is the speed-up table drawn in the
review.

## Head to head with sumaddury's build

`head_to_head/` traced the same 5,000 alphas with this kernel and with the
`metal-port` branch at `e2e72538`. That branch's kernel reads the February
2026 table layout, `(cells*64, fields)`, so `run_ours.py` writes the table in
both layouts. `leak_check.py` measured that branch's per-call memory growth.

To repeat it, build `e2e72538` separately: run CMake with
`-DGIT_SUBMODULE=OFF`, `-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON` and a Boost
include directory. Run `run_theirs.py` from that worktree's root, with the
build's `firm3dpp` first on `PYTHONPATH`.
