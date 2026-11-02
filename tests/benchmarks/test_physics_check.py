import math
import unittest

import numpy as np

import physics_check as phys
from tests.benchmarks.support import SEED, TMAX, final_states, rng

N = 2000


def survive(states: np.ndarray, idx: np.ndarray) -> None:
    """Make the markers idx survive (in place)."""
    states[idx, 0] = TMAX


def lose(states: np.ndarray, idx: np.ndarray) -> None:
    """Make the markers idx lost mid-trace (in place)."""
    states[idx, 0] = 0.5 * TMAX


class TestCompare(unittest.TestCase):
    def setUp(self) -> None:
        self.a = final_states(N, TMAX, 0.05)  # first 100 lost
        self.lost = np.arange(100)
        self.alive = np.arange(100, N)

    def test_identical_traces(self) -> None:
        c = phys.compare(self.a, self.a.copy(), TMAX)
        self.assertTrue(c.identical)
        self.assertEqual((c.only_a, c.only_b), (0, 0))
        self.assertEqual((c.fate_p, c.time_p, c.loss_change), (1.0, 1.0, 0.0))

    def test_symmetric_scatter_passes(self) -> None:
        b = self.a.copy()
        survive(b, self.lost[:6])
        lose(b, self.alive[:6])
        c = phys.compare(self.a, b, TMAX)
        self.assertEqual((c.only_a, c.only_b), (6, 6))
        self.assertEqual(c.fate_p, 1.0)
        self.assertEqual(c.loss_change, 0.0)
        lo, hi = c.loss_change_ci
        self.assertAlmostEqual(lo, -hi)  # symmetric interval about zero

    def test_systematic_extra_losses_are_caught(self) -> None:
        b = self.a.copy()
        lose(b, self.alive[:25])
        c = phys.compare(self.a, b, TMAX)
        self.assertEqual((c.only_a, c.only_b), (0, 25))
        self.assertLess(c.fate_p, 1e-6, f"seed {SEED}")
        self.assertAlmostEqual(c.loss_change, 25 / 100)
        # with every discordant marker on b's side, the upper bound is exact
        self.assertAlmostEqual(c.loss_change_ci[1], 25 / 100)
        self.assertGreater(c.loss_change_ci[0], 0.0)

    def test_systematically_earlier_losses_are_caught(self) -> None:
        b = self.a.copy()
        b[self.lost, 0] *= 0.7
        c = phys.compare(self.a, b, TMAX)
        self.assertEqual((c.only_a, c.only_b), (0, 0))
        self.assertAlmostEqual(c.time_ratio, 0.7)
        self.assertLess(c.time_p, 1e-6, f"seed {SEED}")

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "different ensembles"):
            phys.compare(self.a, self.a[:10], TMAX)


class TestLossChange(unittest.TestCase):
    def test_no_discordant_markers(self) -> None:
        self.assertEqual(phys.loss_change(0, 0, 40), (0.0, (0.0, 0.0)))

    def test_nothing_lost_in_reference_is_undefined(self) -> None:
        change, (lo, hi) = phys.loss_change(0, 3, 0)
        self.assertTrue(math.isnan(change) and math.isnan(lo) and math.isnan(hi))

    def test_interval_contains_estimate(self) -> None:
        change, (lo, hi) = phys.loss_change(5, 3, 44)
        self.assertAlmostEqual(change, -2 / 44)
        self.assertLess(lo, change)
        self.assertGreater(hi, change)


class TestMachineChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.cpu = final_states(N, TMAX, 0.05)

    def test_first_machine_is_the_reference(self) -> None:
        checks = phys.check_machines([(self.cpu, self.cpu), (self.cpu, self.cpu)], TMAX)
        self.assertIsNone(checks[0].gpu_vs_reference)
        self.assertIsNotNone(checks[1].gpu_vs_reference)
        self.assertEqual(len(checks[0].comparisons), 1)
        self.assertEqual(len(checks[1].comparisons), 3)

    def test_bonferroni_threshold(self) -> None:
        checks = phys.check_machines([(self.cpu, self.cpu), (self.cpu, self.cpu)], TMAX)
        self.assertAlmostEqual(checks[0].threshold, phys.ALPHA / 2)
        self.assertAlmostEqual(checks[1].threshold, phys.ALPHA / 6)

    def test_biased_gpu_fails_and_clean_one_passes(self) -> None:
        biased = self.cpu.copy()
        lose(biased, np.arange(100, 130))
        checks = phys.check_machines([(self.cpu, self.cpu), (self.cpu, biased)], TMAX)
        self.assertTrue(checks[0].passed)
        self.assertFalse(checks[1].passed)


class TestTable(unittest.TestCase):
    def test_rows_show_reference_identical_and_failure(self) -> None:
        cpu = final_states(N, TMAX, 0.05)
        biased = cpu.copy()
        lose(biased, np.arange(100, 130))
        checks = phys.check_machines([(cpu, cpu), (cpu, biased)], TMAX)
        text = phys.table(["ref machine", "other machine"], checks)
        ref_row, other_row = [line for line in text.splitlines() if "machine |" in line]
        self.assertIn("reference", ref_row)
        self.assertIn("identical", other_row)  # its CPU matches the reference CPU
        self.assertIn("**FAIL**", other_row)
        self.assertTrue(text.startswith("<!-- markdownlint-disable MD013 -->"))

    def test_describe_agreeing_fates(self) -> None:
        a = final_states(N, TMAX, 0.05)
        b = a.copy()
        b[:100, 0] *= rng(5).uniform(0.9, 1.1, 100)
        self.assertTrue(
            phys.describe(phys.compare(a, b, TMAX)).startswith("fates agree")
        )


if __name__ == "__main__":
    unittest.main()
