import sys
import unittest

import machine as mach
from tests.benchmarks.support import machine_record


class TestLabels(unittest.TestCase):
    def test_label_names_every_distinguishing_field(self) -> None:
        label = mach.label(machine_record())
        self.assertEqual(
            label,
            "MacBook Pro (Mac16,1) · Apple M4 · CPU 10 (4P+6E) · GPU 10-core "
            "· 16 GB · macOS 26.6.2",
        )

    def test_slug_is_filesystem_safe(self) -> None:
        slug = mach.slug(machine_record(chip="Apple M4 Pro", gpu_cores=20))
        self.assertEqual(slug, "Mac16-1_M4Pro-20gpu")
        self.assertNotIn(",", slug)
        self.assertNotIn(" ", slug)

    def test_hardware_key_ignores_macos(self) -> None:
        old = mach.hardware_key(machine_record(macos="26.6.2"))
        new = mach.hardware_key(machine_record(macos="26.7"))
        self.assertEqual(old, new)

    def test_hardware_key_tells_binned_chips_apart(self) -> None:
        a = mach.hardware_key(machine_record(chip="Apple M4 Pro", gpu_cores=16))
        b = mach.hardware_key(machine_record(chip="Apple M4 Pro", gpu_cores=20))
        self.assertNotEqual(a, b)


@unittest.skipUnless(sys.platform == "darwin", "reads macOS hardware and state")
class TestThisMac(unittest.TestCase):
    def test_machine_record_is_complete(self) -> None:
        record = mach.machine()
        self.assertEqual(set(record), set(machine_record()))
        self.assertEqual(
            record["cpu_cores"],
            record["cpu_performance_cores"] + record["cpu_efficiency_cores"],
        )
        self.assertGreater(record["gpu_cores"], 0)

    def test_conditions(self) -> None:
        now = mach.conditions()
        self.assertIn(now["thermal_state"], mach.THERMAL_STATES)
        self.assertIsInstance(now["low_power_mode"], bool)
        self.assertGreaterEqual(now["load_average"], 0.0)


if __name__ == "__main__":
    unittest.main()
