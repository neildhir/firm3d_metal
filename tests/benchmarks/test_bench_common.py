import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import bench_common as bc

TMAX = 1e-2


class TestLossClassification(unittest.TestCase):
    """Survivors that end on float32(tmax) must not count as lost."""

    def test_float32_survivor_is_not_lost(self) -> None:
        final_times = np.array([np.float64(np.float32(TMAX))])
        self.assertLess(final_times[0], TMAX)  # the case the margin exists for
        self.assertFalse(bc.lost_mask(final_times, TMAX)[0])

    def test_survivor_at_tmax_is_not_lost(self) -> None:
        self.assertFalse(bc.lost_mask(np.array([TMAX]), TMAX)[0])

    def test_early_end_is_lost(self) -> None:
        self.assertTrue(bc.lost_mask(np.array([0.5 * TMAX]), TMAX)[0])

    def test_n_lost_counts_traces_ending_early(self) -> None:
        res_tys = [
            np.array([[0.0, 0.5], [t, 0.5]])
            for t in (0.1 * TMAX, np.float32(TMAX), TMAX, 0.9 * TMAX)
        ]
        self.assertEqual(bc.n_lost(res_tys, TMAX), 2)


class TestSnapSurvivorTimes(unittest.TestCase):
    def test_snaps_float32_survivors_and_leaves_losses(self) -> None:
        survivor = np.array([[0.0, 1.0], [np.float32(TMAX), 2.0]])
        lost = np.array([[0.0, 1.0], [0.5 * TMAX, 2.0]])
        bc.snap_survivor_times_inplace([survivor, lost], TMAX)
        self.assertEqual(survivor[-1, 0], TMAX)
        self.assertEqual(lost[-1, 0], 0.5 * TMAX)
        self.assertEqual(survivor[-1, 1], 2.0)  # only the time moves


class TestWriteResults(unittest.TestCase):
    def test_json_ends_with_newline_and_states_round_trip(self) -> None:
        states = np.arange(15, dtype=np.float64).reshape(3, 5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cpu.json"
            bc.write_results(path, {"backend": "cpu"}, states)
            text = path.read_text()
            self.assertTrue(text.endswith("\n"))
            self.assertEqual(json.loads(text), {"backend": "cpu"})
            stored = np.load(bc.final_states_path(path))["final_states"]
        np.testing.assert_array_equal(stored, states)

    def test_final_states_sit_beside_the_json(self) -> None:
        path = Path("/x/metal.json")
        self.assertEqual(bc.final_states_path(path), Path("/x/metal_final_states.npz"))


if __name__ == "__main__":
    unittest.main()
