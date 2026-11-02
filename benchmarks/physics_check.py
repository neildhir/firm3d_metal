"""Marker-by-marker physics comparison of benchmark traces.

The sanity check in run_benchmark.py compares loss *counts*, with a bound
loose enough that only a broken install fails it. Every trace of a benchmark
follows the same seeded markers, though, so two traces (CPU vs GPU on one
machine, or one machine's GPU vs another's) can be compared marker by
marker, which is far more sensitive:

- **Fate.** Markers lost in one trace but not the other (discordant pairs)
  are expected near the loss boundary, where orbits are chaotic and
  roundoff decides. Without a systematic difference they split evenly
  between the two directions; McNemar's exact test asks whether they do.
- **Loss time.** For markers lost in both traces, a Wilcoxon signed-rank
  test on log(t_b / t_a) asks whether one trace loses them systematically
  sooner.

A machine fails when any of its tests rejects at the family-wise level ALPHA
(Bonferroni over that machine's tests). Symmetric scatter passes however
large it is, since it does not bias loss statistics; the number of
discordant markers is reported so that unusually noisy traces stay visible.

Power is limited by how few markers are lost. In the alpha-loss-wistell-a
benchmark about 44 of 5,000 are, and a handful end differently between
CPU and GPU, so a GPU must lose some 35-40% more markers before the check
flags it. A pass therefore means "no large systematic difference", not
"the same physics to a few percent". What the data do support is given by
a CONFIDENCE interval on the relative change in the number of markers
lost (b against a): exact (Clopper-Pearson) on how the discordant markers
split, conditional on their number. Tighter claims need more lost
markers: a larger ensemble or a longer tmax.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, wilcoxon

from bench_common import lost_mask

ALPHA = 0.01  # family-wise false-alarm rate of one machine's check
CONFIDENCE = 0.95  # of the interval on the relative change in losses
TESTS_PER_COMPARISON = 2  # fate and loss time


@dataclass(frozen=True)
class Comparison:
    """Marker-by-marker comparison of two traces, a and b, of one ensemble."""

    identical: bool  # final states bitwise equal
    only_a: int  # markers lost in a only
    only_b: int  # markers lost in b only
    n_both: int  # markers lost in both
    loss_change: float  # relative change in markers lost, b against a
    loss_change_ci: tuple[float, float]  # its CONFIDENCE interval
    time_ratio: float  # median loss-time ratio t_b / t_a, markers lost in both
    fate_p: float  # McNemar exact p-value; 1 without discordant markers
    time_p: float  # Wilcoxon signed-rank p-value; 1 without differing times

    @property
    def min_p(self) -> float:
        return min(self.fate_p, self.time_p)


@dataclass(frozen=True)
class MachineCheck:
    """All comparisons of one machine's traces, and whether they pass.

    ``backends`` compares this machine's CPU (a) with its GPU (b); the
    ``*_vs_reference`` comparisons set the reference machine's trace as a
    and this machine's as b, and are None for the reference itself.
    """

    backends: Comparison
    gpu_vs_reference: Comparison | None
    cpu_vs_reference: Comparison | None

    @property
    def comparisons(self) -> list[Comparison]:
        return [
            c
            for c in (self.backends, self.gpu_vs_reference, self.cpu_vs_reference)
            if c is not None
        ]

    @property
    def threshold(self) -> float:
        """Per-test p-value threshold (Bonferroni)."""
        return ALPHA / (TESTS_PER_COMPARISON * len(self.comparisons))

    @property
    def passed(self) -> bool:
        return all(c.min_p >= self.threshold for c in self.comparisons)


def check_machines(
    traces: list[tuple[np.ndarray, np.ndarray]], tmax: float
) -> list[MachineCheck]:
    """Check every machine, the first serving as reference for the others.

    Parameters
    ----------
    traces : list[tuple[np.ndarray, np.ndarray]]
        Per machine, the (cpu, gpu) final states of the validation
        ensemble, each (nparticles, 5) float64 (t, s, theta, zeta, vpar).
    tmax : float
        Tracing time of the validation ensemble [s].

    Returns
    -------
    list[MachineCheck]
        One per machine, in the order given.
    """
    ref_cpu, ref_gpu = traces[0]
    return [
        MachineCheck(
            backends=compare(cpu, gpu, tmax),
            gpu_vs_reference=None if i == 0 else compare(ref_gpu, gpu, tmax),
            cpu_vs_reference=None if i == 0 else compare(ref_cpu, cpu, tmax),
        )
        for i, (cpu, gpu) in enumerate(traces)
    ]


def compare(a: np.ndarray, b: np.ndarray, tmax: float) -> Comparison:
    """Compare two traces of the same markers, marker by marker.

    Parameters
    ----------
    a, b : np.ndarray
        (nparticles, 5) float64 final states (t, s, theta, zeta, vpar) of
        the same ensemble, markers in the same order.
    tmax : float
        Tracing time [s].

    Returns
    -------
    Comparison
        Discordant fates, loss-time shift, and both tests' p-values.

    Raises
    ------
    ValueError
        If the traces cover different numbers of markers.
    """
    if a.shape != b.shape:
        raise ValueError(f"traces of different ensembles: {a.shape} vs {b.shape}")
    lost_a, lost_b = lost_mask(a[:, 0], tmax), lost_mask(b[:, 0], tmax)
    only_a = int(np.sum(lost_a & ~lost_b))
    only_b = int(np.sum(~lost_a & lost_b))
    both = lost_a & lost_b
    log_ratio = np.log(b[both, 0] / a[both, 0])
    shifted = log_ratio[log_ratio != 0.0]  # identical loss times carry no sign
    time_p = 1.0
    if shifted.size:
        result = wilcoxon(shifted)
        # scipy's stubs do not declare the result's pvalue attribute
        time_p = float(result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]
    change, change_ci = loss_change(only_a, only_b, only_a + int(both.sum()))
    return Comparison(
        identical=bool(np.array_equal(a, b)),
        only_a=only_a,
        only_b=only_b,
        n_both=int(both.sum()),
        loss_change=change,
        loss_change_ci=change_ci,
        time_ratio=float(np.exp(np.median(log_ratio))) if log_ratio.size else 1.0,
        fate_p=binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0,
        time_p=time_p,
    )


def loss_change(
    only_a: int, only_b: int, lost_a: int
) -> tuple[float, tuple[float, float]]:
    """Relative change in markers lost, b against a, with its interval.

    Only discordant markers change the count: b loses
    (only_b - only_a) / lost_a more, relatively. Conditional on there being
    n = only_a + only_b of them, only_b is binomial with an unknown share
    pi, and the change is n (2 pi - 1) / lost_a, so an exact interval on pi
    maps to one on the change.

    Parameters
    ----------
    only_a, only_b : int
        Markers lost in trace a only, and in trace b only.
    lost_a : int
        Markers lost in trace a.

    Returns
    -------
    tuple[float, tuple[float, float]]
        The relative change and its CONFIDENCE interval; zero when no
        marker's fate differs, NaN when trace a lost none.
    """
    discordant = only_a + only_b
    if discordant == 0:
        return 0.0, (0.0, 0.0)
    if lost_a == 0:
        return float("nan"), (float("nan"), float("nan"))
    share = binomtest(only_b, discordant).proportion_ci(CONFIDENCE, method="exact")
    scale = discordant / lost_a
    return (only_b - only_a) / lost_a, (
        scale * (2 * share.low - 1),
        scale * (2 * share.high - 1),
    )


def table(labels: list[str], checks: list[MachineCheck]) -> str:
    """Markdown table of the marker-by-marker physics check.

    Each cell compares two traces of the validation ensemble: the markers
    whose fate differs (lost in the first only / the second only) with
    McNemar's p, the relative change in markers lost (second against
    first) with its CONFIDENCE interval, and the median loss-time ratio
    (second / first) of markers lost in both with the Wilcoxon signed-rank
    p. Wrapped in markdownlint comments lifting the line-length rule: the
    full machine labels do not fit in 80 columns.
    """
    rows = [
        "<!-- markdownlint-disable MD013 -->",
        "| Machine | CPU vs GPU | Reference GPU vs GPU "
        "| Reference CPU vs CPU | Check |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, check in zip(labels, checks, strict=True):
        cells = [
            "reference" if c is None else describe(c)
            for c in (check.backends, check.gpu_vs_reference, check.cpu_vs_reference)
        ]
        verdict = "pass" if check.passed else f"**FAIL** (p < {check.threshold:.2g})"
        rows.append(f"| {label} | {' | '.join(cells)} | {verdict} |")
    rows.append("<!-- markdownlint-enable MD013 -->")
    return "\n".join(rows)


def describe(c: Comparison) -> str:
    """One physics-table cell: discordant fates and loss-time shift."""
    if c.identical:
        return "identical"
    fates = c.only_a + c.only_b
    if fates == 0:
        return f"fates agree; loss time ×{c.time_ratio:.3f}, p {c.time_p:.2g}"
    return (
        f"{fates} fates differ ({c.only_a} / {c.only_b}), p {c.fate_p:.2g}; "
        f"losses {c.loss_change:+.0%} ({CONFIDENCE:.0%} CI "
        f"{c.loss_change_ci[0]:+.0%} to {c.loss_change_ci[1]:+.0%}); "
        f"loss time ×{c.time_ratio:.3f}, p {c.time_p:.2g}"
    )
