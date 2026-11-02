import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import plot_common as pc
from tests.benchmarks.support import machine_record, write_loss_run


def run(
    name: str,
    created: str,
    tag: str = "neil",
    sanity: bool = True,
    metal: str = "m0",
    config: str = "c0",
    **machine: object,
) -> pc.Run:
    record = {
        "created": created,
        "tag": tag,
        "machine": machine_record(**machine),
        "sanity": {"passed": sanity},
        "fingerprints": {"config": config, "equilibrium": "e0", "metal_sources": metal},
    }
    return pc.Run(Path(name), record, {}, {})


def quietly(function, *args):
    """Call function, returning its result and what it warned on stderr."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        result = function(*args)
    return result, err.getvalue()


class TestSelectSeries(unittest.TestCase):
    def test_latest_run_per_machine_with_first_run_date(self) -> None:
        runs = [
            run("old", "2026-11-01T10:00:00+00:00", macos="26.6"),
            run("new", "2026-11-03T10:00:00+00:00", macos="26.7"),
        ]
        (series,) = pc.select_series(runs)
        self.assertEqual(series.run.folder.name, "new")
        self.assertIn("macOS 26.7", series.label)
        self.assertEqual(
            series.first_run, datetime.fromisoformat("2026-11-01T10:00:00+00:00")
        )

    def test_line_styles_follow_first_runs(self) -> None:
        runs = [
            run("m4", "2026-11-01T10:00:00+00:00"),
            run("m1", "2026-11-02T10:00:00+00:00", chip="Apple M1", gpu_cores=7),
            run("m4-again", "2026-11-05T10:00:00+00:00"),
        ]
        series = pc.select_series(runs)
        self.assertEqual([s.linestyle for s in series], ["-", "--"])
        self.assertEqual(series[0].run.folder.name, "m4-again")

    def test_failed_runs_are_skipped_with_a_warning(self) -> None:
        runs = [
            run("ok", "2026-11-01T10:00:00+00:00"),
            run("bad", "2026-11-02T10:00:00+00:00", tag="x", sanity=False),
        ]
        series, warned = quietly(pc.select_series, runs)
        self.assertEqual([s.run.folder.name for s in series], ["ok"])
        self.assertIn("bad", warned)

    def test_fifth_machine_is_an_error(self) -> None:
        runs = [
            run(f"r{i}", f"2026-11-0{i + 1}T10:00:00+00:00", gpu_cores=10 + i)
            for i in range(5)
        ]
        with self.assertRaisesRegex(ValueError, "5 machines"):
            pc.select_series(runs)

    def test_identical_hardware_is_told_apart_by_tag(self) -> None:
        runs = [
            run("a", "2026-11-01T10:00:00+00:00", tag="neil"),
            run("b", "2026-11-02T10:00:00+00:00", tag="colleague"),
        ]
        labels = [s.label for s in pc.select_series(runs)]
        self.assertTrue(
            labels[0].endswith("[neil]") and labels[1].endswith("[colleague]")
        )

    def test_different_metal_sources_warn(self) -> None:
        runs = [
            run("a", "2026-11-01T10:00:00+00:00"),
            run("b", "2026-11-02T10:00:00+00:00", tag="x", metal="m1"),
        ]
        _, warned = quietly(pc.select_series, runs)
        self.assertIn("Metal kernel sources", warned)


class TestCheckComparable(unittest.TestCase):
    def test_different_workloads_are_refused(self) -> None:
        runs = [
            run("a", "2026-11-01T10:00:00+00:00"),
            run("b", "2026-11-02T10:00:00+00:00", config="c1"),
        ]
        with self.assertRaisesRegex(ValueError, "different workloads"):
            pc.check_comparable(runs)

    def test_same_workload_passes(self) -> None:
        pc.check_comparable(
            [
                run("a", "2026-11-01T10:00:00+00:00"),
                run("b", "2026-11-02T10:00:00+00:00"),
            ]
        )


class TestLoadRuns(unittest.TestCase):
    def test_skips_incomplete_runs_and_sorts_by_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp)
            write_loss_run(bench, "z_first", created="2026-11-01T10:00:00+00:00")
            write_loss_run(bench, "a_second", created="2026-11-02T10:00:00+00:00")
            (bench / "crashed.incomplete").mkdir()
            (bench / "no_record").mkdir()
            runs, warned = quietly(pc.load_runs, bench)
        self.assertEqual([r.folder.name for r in runs], ["z_first", "a_second"])
        self.assertIn("crashed.incomplete", warned)
        self.assertIn("no_record", warned)
        self.assertEqual(json.dumps(runs[0].cpu)[:1], "{")


if __name__ == "__main__":
    unittest.main()
