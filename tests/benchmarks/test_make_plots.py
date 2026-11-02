import contextlib
import io
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock


import make_plots as mp
from plot_common import Headline
from tests.benchmarks.support import (
    N,
    TMAX,
    final_states,
    machine_record,
    write_conservation_run,
    write_loss_run,
)

PAGE = (
    "intro\n<!-- make_plots:summary -->\nstale\n<!-- /make_plots:summary -->\noutro\n"
)


class TestFillPage(unittest.TestCase):
    def test_replaces_between_markers_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "b.md"
            page.write_text(PAGE)
            mp.fill_page(page, {"summary": "| table |"})
            text = page.read_text()
        self.assertEqual(
            text,
            "intro\n<!-- make_plots:summary -->\n| table |\n"
            "<!-- /make_plots:summary -->\noutro\n",
        )

    def test_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "b.md"
            page.write_text(PAGE)
            mp.fill_page(page, {"summary": "| table |"})
            once = page.read_text()
            mp.fill_page(page, {"summary": "| table |"})
            self.assertEqual(page.read_text(), once)

    def test_missing_markers_or_page_only_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "b.md"
            page.write_text(PAGE)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                mp.fill_page(page, {"physics": "| table |"})
                mp.fill_page(Path(tmp) / "missing.md", {"summary": "| t |"})
            self.assertEqual(page.read_text(), PAGE)
        self.assertIn("make_plots:physics", err.getvalue())
        self.assertIn("missing.md", err.getvalue())


class TestGlanceTable(unittest.TestCase):
    def test_rows_by_first_run_and_dashes_for_missing_benchmarks(self) -> None:
        early = datetime.fromisoformat("2026-11-01T10:00:00+00:00")
        late = datetime.fromisoformat("2026-11-03T10:00:00+00:00")
        loss: dict[tuple[str | int, ...], Headline] = {
            ("a",): Headline(late, "Mac A", {"Loss": "6.7×"})
        }
        conservation: dict[tuple[str | int, ...], Headline] = {
            ("a",): Headline(late, "Mac A", {"Drift": "2e-6"}),
            ("b",): Headline(early, "Mac B", {"Drift": "3e-6"}),
        }
        lines = mp.glance_table([loss, conservation]).splitlines()
        self.assertEqual(lines[1], "| Machine | Loss | Drift |")
        self.assertEqual(lines[3], "| Mac B | — | 3e-6 |")
        self.assertEqual(lines[4], "| Mac A | 6.7× | 2e-6 |")
        self.assertEqual(lines[0], "<!-- markdownlint-disable MD013 -->")


class TestEndToEnd(unittest.TestCase):
    """make_plots.main over synthetic runs of both kinds and two machines."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.results, self.plots = root / "results", root / "plots"
        m1 = machine_record(
            chip="Apple M1", gpu_cores=7, model_identifier="MacBookAir10,1"
        )
        loss = self.results / "loss-bench"
        write_loss_run(loss, "m4", created="2026-11-01T10:00:00+00:00")
        biased = final_states(N, TMAX, 0.1)
        biased[N // 10 : N // 10 + 40, 0] = 0.5 * TMAX  # 40 extra GPU losses
        write_loss_run(
            loss,
            "m1",
            created="2026-11-02T10:00:00+00:00",
            tag="c",
            machine=m1,
            gpu_states=biased,
        )
        cons = self.results / "cons-bench"
        write_conservation_run(cons, "m4", created="2026-11-01T11:00:00+00:00")
        write_conservation_run(  # a Mac whose CPU leg was memory-capped
            cons,
            "m1",
            created="2026-11-02T11:00:00+00:00",
            tag="c",
            machine=m1,
            mpi_ranks=5,
        )
        for page, markers in (
            ("loss-bench.md", ("summary", "physics")),
            ("cons-bench.md", ("summary", "agreement")),
            ("README.md", ("at-a-glance",)),
        ):
            text = "".join(
                f"<!-- make_plots:{m} -->\n<!-- /make_plots:{m} -->\n" for m in markers
            )
            (root / page).write_text(text)
        self.root = root
        err, out = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(mp, "RESULTS_DIR", self.results),
            mock.patch.object(mp, "PLOTS_DIR", self.plots),
            mock.patch.object(mp, "BENCH_DIR", root),
            contextlib.redirect_stderr(err),
            contextlib.redirect_stdout(out),
        ):
            mp.main([])
        self.warned = err.getvalue()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_writes_every_figure(self) -> None:
        for figure in (
            "loss-bench/scaling.png",
            "loss-bench/validation.png",
            "loss-bench/gpu_machines.png",
            "cons-bench/drift_vs_tolerance.png",
            "cons-bench/work_precision.png",
        ):
            self.assertGreater((self.plots / figure).stat().st_size, 10_000, figure)

    def test_fills_every_page(self) -> None:
        loss_page = (self.root / "loss-bench.md").read_text()
        self.assertIn("Apple M1", loss_page)
        self.assertIn("**FAIL**", loss_page)  # the biased GPU
        cons_page = (self.root / "cons-bench.md").read_text()
        self.assertIn("identical", cons_page)  # same synthetic drifts on both machines
        glance = (self.root / "README.md").read_text()
        self.assertIn("Alpha losses: speedup", glance)
        self.assertIn("Conservation: best median drift", glance)
        self.assertEqual(glance.count("| MacBook"), 2)
        self.assertEqual(glance.count("memory-capped"), 1)  # only the capped Mac

    def test_flags_the_failed_physics_check(self) -> None:
        self.assertIn("physics check FAILED", self.warned)


if __name__ == "__main__":
    unittest.main()
