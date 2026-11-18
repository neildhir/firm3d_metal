# M4 runs of 17 November 2026, before the particle was pinned

Two runs on the M4 (Mac16,1, 10-core GPU) at commit `9a768a59`, which skips
finished particles in the Metal trace kernel's DP5 stages:

- `alpha-loss-wistell-a/2026-11-17-1418_Mac16-1_M4-10gpu_neildhir`
- `conservation-qh/2026-11-17-1424_Mac16-1_M4-10gpu_neildhir`

They traced firm3d's alpha mass as upstream changed it on 2026-10-31
(merged here on 2026-11-13), while their configs and fingerprints are those
of the earlier runs under the same benchmark names, which traced two proton
plus two neutron masses. Filed there, they would mix two workloads under one
fingerprint, so they are kept here and never plotted. Their traced particle
is the one `alpha-loss-wistell-a-v2` and `conservation-qh-v2` pin.

Against the 12 November M4 runs, their Metal legs took 1.47-1.51x less wall
time for alpha losses at 5,000-10,000 particles, and 1.73-1.81x less in the
tol 1e-10 conservation cells. The kernel change accounts for most of that,
but the two sets of runs differ in the alpha's mass too.
