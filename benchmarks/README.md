# CPU vs Metal GPU tracing benchmarks

FIRM3D's production MPI CPU tracer (float64) against the new Metal GPU
backend (float32), on Apple-Silicon Macs and on identical physics. Each
benchmark asks two questions: how much faster is the GPU than all CPU
cores of the same Mac combined, and is its physics output the same? Runs
from every contributor's Mac are filed here and plotted together.

## At a glance

<!-- make_plots:at-a-glance -->
<!-- markdownlint-disable MD013 -->
| Machine | Alpha losses: speedup, 5,000 × 10 ms | Alpha losses: GPU vs CPU markers lost (95% CI) | Conservation: best median drift, CPU / GPU | Conservation: speedup at the GPU's best (with builds) |
| --- | --- | --- | --- | --- |
| MacBook Pro (Mac16,1) · Apple M4 · CPU 10 (4P+6E) · GPU 10-core · 16 GB · macOS 26.6.2 | 6.7× | -5% (-15% to +9%) | 1.7e-07 / 1.9e-06 | 4.7× (1.1×) |
<!-- markdownlint-enable MD013 -->
<!-- /make_plots:at-a-glance -->

Speedups are wall-time ratios, CPU (all cores, MPI) over GPU. The loss
interval is the 95% range of the GPU's change in markers lost, relative
to the CPU. Drift is the change of a conserved momentum, normalised as on
the [conservation page](conservation-qh.md); "with builds" charges each
backend's one-off field build to a single trace, the worst case.

## Benchmarks

| Benchmark | What it measures |
| --- | --- |
| [Alpha losses in Wistell-A](alpha-loss-wistell-a.md) | speed and loss statistics |
| [p_η conservation, QH field](conservation-qh.md) | accuracy against a ground truth |

The machine label in every table and legend reads: model (identifier) ·
chip · CPU cores (performance + efficiency) · GPU cores · memory · macOS.
In the figures, colour is the backend (blue CPU, orange GPU) and line
style the machine (solid, dashed, dotted, dash-dot, in order of each
machine's first run); lines are semi-transparent so coincident curves show
through, and each machine is drawn from its latest run that passed the
run-time sanity check.

## Benchmark your Mac

Contributions from any M-series Mac are welcome.

**1. Install** (about 3 minutes; asks before installing anything with
Homebrew):

```sh
git clone https://github.com/neildhir/firm3d_metal.git
cd firm3d_metal
bash install/macOS/install_firm3d_metal.sh
```

**2. Run both benchmarks** (about 6 and 9 minutes on a base M4). Plug in
the power adapter, quit heavy apps, and leave the machine alone until they
finish:

```sh
cd benchmarks
../.venv/bin/python run_benchmark.py alpha-loss-wistell-a.yaml --tag <you>
../.venv/bin/python run_benchmark.py conservation-qh.yaml --tag <you>
```

where `<you>` is your GitHub handle. Each run refuses to start on
battery, in Low Power Mode, on a stale build, or with uncommitted changes,
and ends by printing your numbers and either "Sanity check passed" or why
not. A failed run must not be submitted: report it instead.

**3. Submit** — see [Submitting a run](#submitting-a-run).

## Submitting a run

Results reach this repository by pull request.

1. Fork `neildhir/firm3d_metal` on GitHub, then add your fork as a remote:
   `git remote add fork https://github.com/<you>/firm3d_metal.git`.
2. Commit **only your run folders**, on a branch:

   ```sh
   git switch -c results/<you>
   git add benchmarks/results/*/<your-run-folders>
   git commit -m "chore: add <chip> benchmark runs"
   git push fork results/<you>
   ```

3. Open a pull request into **`neildhir/firm3d_metal`**, branch `master`.
   Check the base repository before submitting: this fork is itself a
   fork of `ColumbiaStellaratorTheory/firm3d`, and GitHub's pull-request
   form can pre-select that upstream repository instead, which would
   send the unfinished Metal work there. Either open this pre-filled
   link, with your handle in place of `<you>`:

   ```text
   https://github.com/neildhir/firm3d_metal/compare/master...<you>:results/<you>
   ```

   or use the GitHub CLI, which names the target explicitly:

   ```sh
   gh pr create --repo neildhir/firm3d_metal --base master \
       --head <you>:results/<you>
   ```

Rules that keep pull requests conflict-free:

- **Never commit `plots/` or the Markdown pages.** The maintainer
  regenerates both after merging with `python make_plots.py`, which
  redraws every figure and rewrites the tables in these pages, so
  contributions never collide.
- Never edit another run's folder, and delete any `*.incomplete` folder
  (a crashed run) rather than submitting it.
- Re-running on the same machine (after an update, say) is welcome: the
  plots show each machine's latest run, the older ones stay on record.

No GitHub account? Zip the run folders and send them to the maintainer.

## Reproducibility and noise

- **Physics results are deterministic.** Seeds fix the markers, and each
  marker integrates independently (no cross-particle coupling, no
  order-dependent reductions on either backend), so re-running a
  benchmark reproduces its results bit for bit on both backends (checked
  on the M4: loss final states and all 160,000 conservation drifts) — on
  the CPU side independently of the number of ranks.
- **Wall times are the noisy part, and macOS's thermal state does not
  see it.** Back-to-back M4 runs agreed to ~1% on the 3-minute loss
  validation leg, but short legs swung by up to 15% (CPU, 10,000 markers ×
  1 ms: 37.7 s, then 32.0 s) and CPU conservation tracing at 64³ by up to
  2×, all with the thermal state "nominal". Each run therefore records
  the load average too, and the run order is fixed (CPU, 60 s cool-down,
  GPU). **Treat speedup differences between machines under ~15% as noise**
  until each machine has repeat runs. Fanless machines (MacBook Air)
  throttle under sustained load; that is a property of the machine, not
  noise, which is why the chassis is in the label.
- **The run-time sanity check catches broken installs, not subtle
  physics.** It compares CPU and GPU loosely (loss counts within 3σ;
  conservation drifts within 10× where float32 resolves the tolerance).
  Claims about physics rest on each benchmark page's own comparisons.
- **Comparability is enforced.** `make_plots.py` refuses to plot runs whose
  workload (config or equilibrium) differs, and warns when the Metal
  sources differ; a config's `execution` section (how the legs run) is not
  part of the workload. `run_benchmark.py` refuses to run with any
  `FIRM3D_METAL_*` debug variable set (`FIRM3D_METAL_FAST_MATH` would shift
  trajectories at roundoff level). To change a workload, copy its config
  to a new benchmark name: its runs and plots then live under that name.

## Files

| File | Holds / does |
| --- | --- |
| `<benchmark>.yaml` | a benchmark's workload (and its `kind`) |
| `<benchmark>.md` | its page, tables filled in by `make_plots.py` |
| `run_benchmark.py` | the one command: checks, both legs, filing |
| `kinds.py` | per kind: leg scripts, sanity check, summary |
| `machine.py` | hardware fingerprint and run conditions |
| `bench_cpu_scaling.py`, `bench_metal_gpu.py` | loss legs |
| `bench_cpu_conservation.py`, `bench_metal_conservation.py` | p_η legs |
| `bench_common.py`, `bench_conservation_common.py` | shared leg code |
| `make_plots.py` | figures and page tables from every run |
| `plot_common.py`, `plots_conservation.py` | plotting by kind |
| `physics_check.py` | marker-by-marker loss comparison |
| `results/<benchmark>/` | one folder per run |
| `results/archive/` | first M4 runs, pre-fingerprint; not plotted |

A run folder holds `run.json` (machine, conditions, fingerprints, sanity
check), `cpu.json` and `metal.json` (per-backend results), and per-marker
data in `cpu_*.npz` and `metal_*.npz`.

To run a leg by hand (for profiling, say), call its script directly from
this directory, e.g.

```sh
../.venv/bin/python bench_metal_gpu.py alpha-loss-wistell-a.yaml metal.json
mpiexec -n <physical cores> ../.venv/bin/python -m mpi4py \
    bench_cpu_scaling.py alpha-loss-wistell-a.yaml cpu.json
```

Always launch an MPI script through `python -m mpi4py`: it converts an
uncaught exception on one rank into `MPI_Abort`, where a plain script would
leave the other ranks deadlocked in their next collective.
