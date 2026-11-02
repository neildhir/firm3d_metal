# Archived: first M4 results (2026-10-27)

The original CPU-vs-Metal runs on one Apple M4 MacBook Pro (Mac16,1,
4P+6E CPU, 10-core GPU, 16 GB), made before `run_benchmark.py` existed.
They carry no machine record or fingerprints, so `make_plots.py` never
reads this folder; they are kept as the evidence cited in the benchmark
README.

- `*_results_tol6.json`: tol 1e-6, the first headline numbers. Config:
  today's `benchmarks/alpha-loss-wistell-a.yaml`, then named
  `bench_scaling_config_tol6.yaml` (the `description` key came later).
- `cpu_results.json`, `metal_results.json`: tol 1e-8, same physics but
  ~2.3× slower on the GPU. Config: `bench_scaling_config.yaml` here.
- `plots/`: the figures from the tol 1e-6 pair.

The JSONs hold the final states inline, the pre-`.npz` format. Both
configs' `boozmn_file` paths are relative to `benchmarks/`.
