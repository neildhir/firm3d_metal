import unittest
from pathlib import Path

import numpy as np

import plots_conservation as pcons
from plot_common import Run


def cell(res: int, tol: float, wall: float, drift: float, build: float = 1.0) -> dict:
    return {
        "resolution": res,
        "tol": tol,
        "wall_s": wall,
        "field_build_s": build,
        "drift": {"p50": drift},
    }


def conservation_run(cpu_cells: list[dict], gpu_cells: list[dict]) -> Run:
    return Run(Path("r"), {}, {"cells": cpu_cells}, {"cells": gpu_cells})


class TestFront(unittest.TestCase):
    def test_keeps_only_cells_nothing_beats(self) -> None:
        times = np.array([1.0, 2.0, 3.0, 4.0])
        drifts = np.array([1e-3, 2e-3, 1e-5, 1e-4])  # (2 s) and (4 s) are beaten
        self.assertEqual(pcons.front(times, drifts), ([1.0, 3.0], [1e-3, 1e-5]))

    def test_unsorted_input(self) -> None:
        xs, ys = pcons.front(np.array([3.0, 1.0]), np.array([1e-6, 1e-3]))
        self.assertEqual((xs, ys), ([1.0, 3.0], [1e-3, 1e-6]))


class TestAtGpuBest(unittest.TestCase):
    def setUp(self) -> None:
        self.gpu = [
            cell(32, 1e-6, 0.4, 1.5e-5, build=3.0),
            cell(32, 1e-8, 0.8, 2e-6, build=3.0),
        ]

    def test_compares_fastest_cells_reaching_the_gpus_best(self) -> None:
        cpu = [
            cell(16, 1e-8, 3.0, 1.5e-6, build=1.0),  # reaches 2e-6, fastest
            cell(64, 1e-10, 20.0, 1e-7, build=8.0),  # the CPU's best
        ]
        cmp = pcons.at_gpu_best(conservation_run(cpu, self.gpu))
        self.assertEqual(cmp.best_cpu["drift"]["p50"], 1e-7)
        self.assertEqual(cmp.cpu, cpu[0])
        self.assertEqual(cmp.gpu, self.gpu[1])
        speedup, with_builds = cmp.speedup, cmp.speedup_with_builds
        if speedup is None or with_builds is None:
            self.fail("the CPU reaches the GPU's best, so both speedups exist")
        self.assertAlmostEqual(speedup, 3.0 / 0.8)
        self.assertAlmostEqual(with_builds, (1.0 + 3.0) / (3.0 + 0.8))

    def test_cpu_that_never_reaches_it(self) -> None:
        cpu = [cell(16, 1e-6, 1.0, 1.6e-5)]
        cmp = pcons.at_gpu_best(conservation_run(cpu, self.gpu))
        self.assertIsNone(cmp.cpu)
        self.assertIsNone(cmp.speedup)
        columns = pcons.glance_columns(cmp, "10")
        self.assertIn("—", columns.values())


class TestCpuRanks(unittest.TestCase):
    """Fewer CPU ranks flatter the GPU, so a memory cap is always shown."""

    def ranks_run(self, ranks: int, cores: int) -> Run:
        record = {"mpi_ranks": ranks, "machine": {"cpu_cores": cores}}
        return Run(Path("r"), record, {}, {})

    def test_uncapped(self) -> None:
        self.assertEqual(pcons.cpu_ranks(self.ranks_run(10, 10)), "10")

    def test_capped(self) -> None:
        ranks = pcons.cpu_ranks(self.ranks_run(5, 8))
        self.assertEqual(ranks, "5 of 8 cores, memory-capped")

    def test_capped_speedup_is_marked_in_the_glance_table(self) -> None:
        gpu = [cell(32, 1e-8, 0.8, 2e-6)]
        cmp = pcons.at_gpu_best(conservation_run([cell(16, 1e-8, 3.0, 1e-6)], gpu))
        speedups = [
            pcons.glance_columns(cmp, ranks)[
                "Conservation: speedup at the GPU's best (with builds)"
            ]
            for ranks in ("10", "5 of 8 cores, memory-capped")
        ]
        self.assertNotIn("CPU on", speedups[0])
        self.assertTrue(speedups[1].endswith("CPU on 5 of 8 cores, memory-capped"))


class TestFormatting(unittest.TestCase):
    def test_cell_name(self) -> None:
        self.assertEqual(pcons.cell_name(cell(64, 1e-8, 1.0, 1e-6)), "64³, tol 1e-8")

    def test_identical_drifts(self) -> None:
        a = np.full((2, 2, 5), 1e-6, dtype=np.float32)
        self.assertEqual(pcons.compare_drifts(a, a.copy()), "identical")

    def test_systematically_larger_drifts(self) -> None:
        a = np.linspace(1e-6, 2e-6, 60).reshape(2, 3, 10)
        text = pcons.compare_drifts(a, 2 * a)
        self.assertTrue(text.startswith("median ratio ×2.000"), text)


if __name__ == "__main__":
    unittest.main()
