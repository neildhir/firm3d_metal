# Benchmark: p_η conservation in a quasi-helical field (v2)

[← All benchmarks](README.md)

The workload of [p_η conservation in a quasi-helical field](conservation-qh.md),
with the traced alpha particle pinned in `conservation-qh-v2.yaml` rather
than taken from firm3d. Upstream firm3d changed the alpha's mass on
2026-10-31, from two proton plus two neutron masses to the measured alpha
mass, 0.76% lighter, which shifts every orbit; this benchmark traces the
new mass. The question, the ground truth, the set-up and the notes are
those on the first benchmark's page.

## Results

<!-- make_plots:summary -->
<!-- markdownlint-disable MD013 -->
| Machine | CPU ranks | Best median drift, CPU | Best median drift, GPU | Tracing to the GPU's best, CPU / GPU | Speedup | With field builds |
| --- | --- | --- | --- | --- | --- | --- |
| MacBook Pro (Mac16,1) · Apple M4 · CPU 10 (4P+6E) · GPU 10-core · 16 GB · macOS 27.2.1 | 10 | 1.7e-07 (64³, tol 1e-10) | 1.9e-06 (32³, tol 1e-8) | 3.42 s (32³, tol 1e-8) / 0.55 s (32³, tol 1e-8) | 6.3× | 1.0× |
| MacBook Pro (MacBookPro18,3) · Apple M1 Pro · CPU 10 (8P+2E) · GPU 14-core · 32 GB · macOS 26.6.2 | 10 | 1.7e-07 (64³, tol 1e-10) | 1.9e-06 (32³, tol 1e-8) | 4.10 s (32³, tol 1e-8) / 0.67 s (32³, tol 1e-8) | 6.2× | 0.8× |
<!-- markdownlint-enable MD013 -->
<!-- /make_plots:summary -->

![p_η drift against tolerance](plots/conservation-qh-v2/drift_vs_tolerance.png)

*Median p_η drift against solver tolerance, one panel per field grid.*

![Accuracy against tracing time](plots/conservation-qh-v2/work_precision.png)

*Accuracy against tracing time: each backend's best accuracy within a given
time.*

## Agreement across machines

<!-- make_plots:agreement -->
<!-- markdownlint-disable MD013 -->
| Machine | GPU drifts vs reference GPU | CPU drifts vs reference CPU |
| --- | --- | --- |
| MacBook Pro (Mac16,1) · Apple M4 · CPU 10 (4P+6E) · GPU 10-core · 16 GB · macOS 27.2.1 | reference | reference |
| MacBook Pro (MacBookPro18,3) · Apple M1 Pro · CPU 10 (8P+2E) · GPU 14-core · 32 GB · macOS 26.6.2 | median ratio ×1.000, p 0.00052 | identical |
<!-- markdownlint-enable MD013 -->
<!-- /make_plots:agreement -->

The comparison is the one described on the
[first benchmark's page](conservation-qh.md#agreement-across-machines).
