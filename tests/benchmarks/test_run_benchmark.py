import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bench_common as bc
import kinds
import run_benchmark as rb
from tests.benchmarks import BENCH_DIR


def config(name: str) -> dict:
    return bc.load_config(BENCH_DIR / f"{name}.yaml")


class TestCpuRanks(unittest.TestCase):
    def test_one_rank_per_core_without_a_memory_hint(self) -> None:
        self.assertEqual(rb.cpu_ranks({"cpu_cores": 10, "memory_gb": 8}, {}), 10)

    def test_memory_caps_the_ranks(self) -> None:
        cfg = {"execution": {"cpu_rank_memory_gb": 1.0}}
        hw = {"cpu_cores": 10, "memory_gb": 8}
        expected = int((8 - rb.MEMORY_RESERVE_GB) // 1.0)
        self.assertEqual(rb.cpu_ranks(hw, cfg), expected)

    def test_never_more_ranks_than_cores(self) -> None:
        cfg = {"execution": {"cpu_rank_memory_gb": 0.1}}
        self.assertEqual(rb.cpu_ranks({"cpu_cores": 10, "memory_gb": 128}, cfg), 10)

    def test_always_at_least_one_rank(self) -> None:
        cfg = {"execution": {"cpu_rank_memory_gb": 64.0}}
        self.assertEqual(rb.cpu_ranks({"cpu_cores": 10, "memory_gb": 8}, cfg), 1)


class TestBuildProblem(unittest.TestCase):
    KERNEL = b"kernel void trace_boozer() {}"

    def problem(self, binary: bytes) -> str | None:
        with tempfile.TemporaryDirectory() as tmp:
            built = Path(tmp) / "firm3dpp.so"
            built.write_bytes(binary)
            return rb.build_problem(str(built), self.KERNEL)

    def test_current_build_passes(self) -> None:
        self.assertIsNone(self.problem(b"head" + self.KERNEL + b"tail"))

    def test_stale_kernel_is_refused(self) -> None:
        problem = self.problem(b"head kernel void trace_boozer_old() {} tail")
        self.assertIn("stale", problem or "")

    def test_openmp_build_is_refused(self) -> None:
        # what a build linking Homebrew's libomp carries in its load commands
        binary = self.KERNEL + b"/opt/homebrew/opt/libomp/lib/libomp.dylib"
        self.assertIn("OpenMP", self.problem(binary) or "")

    def test_unlocated_build_is_refused(self) -> None:
        self.assertIsNotNone(rb.build_problem(None, self.KERNEL))


class TestCpuCommand(unittest.TestCase):
    def test_launches_through_mpi4py(self) -> None:
        cmd = rb.cpu_command(kinds.KINDS["loss"], 4, Path("c.yaml"), Path("o.json"))
        self.assertEqual(cmd[:3], ["mpiexec", "-n", "4"])
        self.assertEqual(
            cmd[-5:], ["-m", "mpi4py", "bench_cpu_scaling.py", "c.yaml", "o.json"]
        )
        self.assertNotIn("-x", cmd)

    def test_passes_the_kinds_environment(self) -> None:
        cmd = rb.cpu_command(
            kinds.KINDS["conservation"], 4, Path("c.yaml"), Path("o.json")
        )
        for var in kinds.KINDS["conservation"].mpi_env:
            i = cmd.index(var)
            self.assertEqual(cmd[i - 1], "-x")


class TestFingerprints(unittest.TestCase):
    def test_key_order_does_not_matter(self) -> None:
        cfg = config("alpha-loss-wistell-a")
        reordered = dict(reversed(list(cfg.items())))
        self.assertEqual(
            rb.fingerprints(cfg)["config"], rb.fingerprints(reordered)["config"]
        )

    def test_execution_section_is_not_part_of_the_workload(self) -> None:
        cfg = config("conservation-qh")
        tuned = dict(cfg, execution={"cpu_rank_memory_gb": 4.0})
        self.assertEqual(
            rb.fingerprints(cfg)["config"], rb.fingerprints(tuned)["config"]
        )

    def test_workload_changes_change_the_fingerprint(self) -> None:
        cfg = config("conservation-qh")
        changed = dict(cfg, nparticles=cfg["nparticles"] + 1)
        self.assertNotEqual(
            rb.fingerprints(cfg)["config"], rb.fingerprints(changed)["config"]
        )

    def test_filed_runs_match_their_benchmark_config(self) -> None:
        """Editing a benchmark's workload in place would orphan its runs.

        Every filed run must still fingerprint like its benchmark's config;
        a changed workload belongs under a new benchmark name.
        """
        runs = sorted((BENCH_DIR / "results").glob("*/*/run.json"))
        runs = [r for r in runs if r.parts[-3] != "archive"]
        self.assertTrue(runs, "no filed runs found")
        for run in runs:
            benchmark = run.parts[-3]
            recorded = json.loads(run.read_text())["fingerprints"]
            current = rb.fingerprints(config(benchmark))
            for key in ("config", "equilibrium"):
                self.assertEqual(
                    recorded[key], current[key], f"{run.parent.name}: {key}"
                )


class TestParseArgs(unittest.TestCase):
    def parse(self, *argv: str):
        with mock.patch("sys.argv", ["run_benchmark.py", *argv]):
            return rb.parse_args()

    def test_valid_tag(self) -> None:
        args = self.parse("conservation-qh.yaml", "--tag", "neil-d2")
        self.assertEqual(
            (args.config, args.tag, args.allow_dirty),
            ("conservation-qh.yaml", "neil-d2", False),
        )

    def test_tag_with_unsafe_characters_is_refused(self) -> None:
        for tag in ("../x", "a b", "-lead"):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                self.parse("conservation-qh.yaml", "--tag", tag)

    def test_tag_is_required(self) -> None:
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            self.parse("conservation-qh.yaml")


if __name__ == "__main__":
    unittest.main()
