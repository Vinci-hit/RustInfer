import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "execution_stats", Path(__file__).resolve().parents[1] / "summarize_execution_stats.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExecutionStatsTests(unittest.TestCase):
    def test_cumulative_summaries_are_not_double_counted(self):
        parsed = module.parse_log('''
INFO execution_stats: execution summary owner="target" phase=Verify calls=10 host_mean_ms=2.5
INFO execution_stats: execution summary owner="target" phase=Verify calls=20 host_mean_ms=3.5
INFO execution_stats: execution summary owner="proposer" phase=Draft calls=20 gpu_mean_ms=0.25
INFO execution_stats: execution tokens owner="target" rounds=20 proposed=0 accepted=0 emitted=20
''')
        self.assertEqual(parsed["target"]["phases"]["Verify"]["calls"], 20)
        self.assertEqual(parsed["proposer"]["phases"]["Draft"]["gpu_mean_ms"], 0.25)
        self.assertNotIn("acceptance_rate", parsed["target"]["tokens"])

    def test_ansi_and_unrelated_lines(self):
        parsed = module.parse_log(
            '\x1b[32mINFO\x1b[0m execution_stats: execution summary owner="target" '
            'phase=Decode calls=1 failures=0\n'
            'DEBUG execution_stats: execution phase owner="target" phase=Decode tokens=1\n'
            'INFO other: execution summary owner="bad" phase=Verify calls=99\n')
        self.assertEqual(set(parsed), {"target"})
        self.assertEqual(parsed["target"]["phases"]["Decode"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
