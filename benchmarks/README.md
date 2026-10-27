# CPU vs Metal GPU tracing benchmarks

Scaling and validation benchmarks comparing FIRM3D's MPI CPU tracer (float64)
against the Metal GPU backend (float32) on Apple Silicon. Both tracers
integrate the same fusion-birth alpha-particle ensembles (same seed, same
sampling) in the same interpolated Wistell-A vacuum equilibrium scaled to
ARIES-CS (`examples/inputs/boozmn_aten_rescaled.nc`).

## Results (Apple M4, 4P + 6E cores, 10-core GPU, 16 GB)

48³ interpolation cells, degree 3, tol 1e-6 (`bench_scaling_config_tol6.yaml`):

| Workload | CPU, 10 MPI ranks | Metal GPU | Speedup |
| --- | --- | --- | --- |
| 10,000 alphas × 1 ms | 33.3 s | 4.8 s | 6.9× |
| 5,000 alphas × 10 ms | 164.8 s | 25.4 s | 6.5× |

Loss fraction after 10 ms: 0.88% (CPU float64) vs 0.84% (Metal float32) —
within Poisson uncertainty of each other (44 vs 42 of 5,000 particles). See
`plots/scaling.png` and `plots/validation.png`.

## Files

| File | Produces / holds |
| --- | --- |
| `bench_scaling_config_tol6.yaml` | tol 1e-6 run → `results/*_tol6.json`, the headline numbers |
| `bench_scaling_config.yaml` | tol 1e-8 run → `results/cpu_results.json`, `results/metal_results.json` |
| `bench_common.py` | shared field build, particle sampling, survivor classification |
| `bench_metal_gpu.py` | GPU benchmark (single process) |
| `bench_cpu_scaling.py` | CPU benchmark (MPI over particles) |
| `make_plots.py` | figures from the two result files |

## Reproducing

From this directory:

```sh
python bench_metal_gpu.py bench_scaling_config_tol6.yaml results/metal_results_tol6.json
mpiexec -n 10 python -m mpi4py bench_cpu_scaling.py bench_scaling_config_tol6.yaml results/cpu_results_tol6.json
python make_plots.py results/cpu_results_tol6.json results/metal_results_tol6.json plots
```

Set `-n` to the machine's physical core count. Always launch the MPI script
through `python -m mpi4py`: it converts an uncaught exception on one rank into
`MPI_Abort`, where a plain script would leave the other ranks deadlocked in
their next collective.

## Numerical notes

- **Use tol 1e-6 for single precision** (the setting of
  `examples/gpu_boozer_tracing`). A tolerance of 1e-8 sits below float32's
  roundoff floor (machine epsilon ≈ 1.2e-7), so the step controller chases
  rounding noise: the tol 1e-8 results in `results/` show the same physics but
  ~2.3× longer GPU wall times.
- **Survivor classification needs a float32-aware margin.** The GPU integrates
  time in float32 and stops at the first step boundary at or after tmax, so a
  survivor can finish exactly on `float32(tmax)` — about 2.2e-9 *below* the
  float64 tmax at tmax = 1e-2. Strict float64 comparisons (including the 1e-15
  margin in `compute_loss_fraction`) then count it as lost;
  `bench_common.snap_survivor_times_inplace` corrects for this before any loss
  accounting.
- `comm` is passed to `BoozerRadialInterpolant` only, never also to
  `InterpolatedBoozerField`: the former parallelises over Fourier modes with
  identically sized collectives on every rank, while point-parallelising the
  latter on top of it nests mismatched Allreduces (`MPI_ERR_TRUNCATE`).
