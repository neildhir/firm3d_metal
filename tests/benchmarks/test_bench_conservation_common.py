import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np

import bench_conservation_common as cc
from bench_common import CHARGE, MASS
from firm3d.field.boozermagneticfield import BoozerAnalytic, BoozerSplineField
from firm3d.trajectory_helpers import compute_peta

HELICITY = {"helicity": {"m": 1, "n": 4}}


def field_with_psi0(psi0: float | None) -> BoozerSplineField:
    """A stand-in carrying only the edge flux, all peta_scale reads."""
    return cast(BoozerSplineField, SimpleNamespace(psi0=psi0))


def analytic_field() -> BoozerAnalytic:
    """A cheap, exactly quasi-helical (M, N) = (1, 4) field."""
    return BoozerAnalytic(etabar=1.0, B0=1.0, N=4, G0=1.1, psi0=0.8, iota0=0.6)


class TestPetaDrift(unittest.TestCase):
    def test_unchanged_state_has_zero_drift(self) -> None:
        row = [0.0, 0.5, 0.1, 0.2, 1e6]
        traces = [np.array([row, row, row])]
        np.testing.assert_array_equal(
            cc.peta_drift(analytic_field(), traces, HELICITY), [0.0]
        )

    def test_drift_is_largest_change_over_scale(self) -> None:
        field = analytic_field()
        trace = np.array(
            [
                [0.0, 0.5, 0.1, 0.2, 1.0e6],
                [1e-5, 0.6, 0.3, 0.1, 0.9e6],
                [2e-5, 0.55, 0.2, 0.15, 0.95e6],
            ]
        )
        peta = compute_peta(
            field,
            np.ascontiguousarray(trace[:, 1:4]),
            trace[:, 4].copy(),
            MASS,
            CHARGE,
            1,
            4,
        )
        expected = np.max(np.abs(peta - peta[0])) / (CHARGE * 0.8)
        (drift,) = cc.peta_drift(field, [trace], HELICITY)
        self.assertGreater(drift, 0.0)
        self.assertAlmostEqual(drift, expected, delta=1e-12 * expected)

    def test_scale_is_charge_times_edge_flux(self) -> None:
        self.assertEqual(cc.peta_scale(field_with_psi0(-2.0)), 2.0 * CHARGE)

    def test_scale_needs_edge_flux(self) -> None:
        with self.assertRaisesRegex(ValueError, "psi0"):
            cc.peta_scale(field_with_psi0(None))


class TestSummarise(unittest.TestCase):
    def test_percentiles_and_max(self) -> None:
        stats = cc.summarise(np.arange(1.0, 101.0))
        self.assertEqual(set(stats), {"p50", "p90", "p99", "max"})
        self.assertAlmostEqual(stats["p50"], 50.5)
        self.assertAlmostEqual(stats["p90"], 90.1)
        self.assertEqual(stats["max"], 100.0)


class TestWriteResults(unittest.TestCase):
    def test_drift_stored_in_float32_and_peta0_in_float64(self) -> None:
        drift = np.full((2, 3, 4), 1.5e-6)
        peta0 = np.linspace(1e-19, 2e-19, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metal.json"
            cc.write_results(path, {"cells": []}, drift, peta0)
            self.assertTrue(path.read_text().endswith("\n"))
            self.assertEqual(json.loads(path.read_text()), {"cells": []})
            stored = np.load(cc.drift_path(path))
            self.assertEqual(stored["drift"].dtype, np.float32)
            self.assertEqual(stored["peta0"].dtype, np.float64)
            np.testing.assert_allclose(stored["drift"], drift, rtol=1e-7)
            np.testing.assert_array_equal(stored["peta0"], peta0)

    def test_drift_file_sits_beside_the_json(self) -> None:
        self.assertEqual(cc.drift_path(Path("/x/cpu.json")), Path("/x/cpu_drift.npz"))


if __name__ == "__main__":
    unittest.main()
