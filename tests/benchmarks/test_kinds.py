import unittest

import kinds
from tests.benchmarks import BENCH_DIR


def loss_leg(scaling_lost: list[int], validation_lost: int) -> dict:
    return {
        "scaling": [
            {"nparticles": 100 * (i + 1), "tmax_s": 1e-3, "wall_s": 1.0, "n_lost": n}
            for i, n in enumerate(scaling_lost)
        ],
        "validation": {
            "nparticles": 5000,
            "tmax_s": 1e-2,
            "wall_s": 10.0,
            "n_lost": validation_lost,
        },
    }


def conservation_leg(drifts: dict[float, float]) -> dict:
    return {
        "cells": [
            {"resolution": 16, "tol": tol, "wall_s": 1.0, "drift": {"p50": d}}
            for tol, d in drifts.items()
        ]
    }


class TestKindOf(unittest.TestCase):
    def test_configs_without_a_kind_are_loss_benchmarks(self) -> None:
        self.assertEqual(kinds.kind_of({}), "loss")

    def test_explicit_kind(self) -> None:
        self.assertEqual(kinds.kind_of({"kind": "conservation"}), "conservation")

    def test_every_kind_names_scripts_that_exist(self) -> None:
        for name, kind in kinds.KINDS.items():
            for script in (kind.cpu_script, kind.metal_script):
                self.assertTrue((BENCH_DIR / script).exists(), f"{name}: {script}")


class TestLossSanity(unittest.TestCase):
    def test_close_counts_pass(self) -> None:
        result = kinds.loss_sanity(loss_leg([1, 7], 44), loss_leg([1, 8], 42))
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 3)

    def test_zero_losses_on_both_pass(self) -> None:
        self.assertTrue(kinds.loss_sanity(loss_leg([0], 0), loss_leg([0], 0))["passed"])

    def test_wildly_different_counts_fail(self) -> None:
        # |44 - 100| = 56 exceeds 3 sqrt(144) = 36
        result = kinds.loss_sanity(loss_leg([1], 44), loss_leg([1], 100))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"][-1]["passed"])


class TestConservationSanity(unittest.TestCase):
    def test_matching_drifts_pass(self) -> None:
        cpu = conservation_leg({1e-4: 3e-3, 1e-6: 1.5e-5})
        gpu = conservation_leg({1e-4: 3e-3, 1e-6: 1.4e-5})
        self.assertTrue(kinds.conservation_sanity(cpu, gpu)["passed"])

    def test_tolerances_below_float32_resolution_are_not_checked(self) -> None:
        cpu = conservation_leg({1e-6: 1.5e-5, 1e-10: 1e-7})
        gpu = conservation_leg({1e-6: 1.5e-5, 1e-10: 5e-6})  # 50x, at the float32 floor
        result = kinds.conservation_sanity(cpu, gpu)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 1)

    def test_too_large_or_too_small_gpu_drift_fails(self) -> None:
        cpu = conservation_leg({1e-6: 1.5e-5})
        for gpu_drift in (1.6e-4, 1.4e-6):  # beyond 10x either way
            gpu = conservation_leg({1e-6: gpu_drift})
            self.assertFalse(kinds.conservation_sanity(cpu, gpu)["passed"], gpu_drift)

    def test_zero_cpu_drift_fails(self) -> None:
        cpu = conservation_leg({1e-6: 0.0})
        gpu = conservation_leg({1e-6: 1e-5})
        self.assertFalse(kinds.conservation_sanity(cpu, gpu)["passed"])


class TestSummaries(unittest.TestCase):
    def test_loss_summary_has_a_line_per_workload(self) -> None:
        lines = kinds.loss_summary(loss_leg([1, 2], 44), loss_leg([1, 2], 42))
        self.assertEqual(len(lines), 1 + 3)
        self.assertIn("44 / 42", lines[-1])

    def test_conservation_summary_has_a_line_per_cell(self) -> None:
        cpu = conservation_leg({1e-4: 3e-3, 1e-6: 1.5e-5})
        lines = kinds.conservation_summary(cpu, cpu)
        self.assertEqual(len(lines), 1 + 2)


if __name__ == "__main__":
    unittest.main()
