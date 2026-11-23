"""CPU checks that portability compatibility cannot admit runtime changes."""
from pathlib import Path
import unittest
from preflight import same_runner_except_paths


class SourceCompatibilityTests(unittest.TestCase):
    def test_path_only_legacy_runner_matches(self):
        current = (Path(__file__).parent/"run.py").read_text()
        legacy = current.replace(
            'os.environ.get("ROBOCURVE_SCHEDULER_ROOT", str(Path.home()/"robocurve-gpu-scheduler"))',
            '"/legacy-user/robocurve-gpu-scheduler"').replace(
            'str(Path.home()/"yam-eval/folding-multitool/runs")',
            '"/legacy-user/yam-eval/folding-multitool/runs"')
        self.assertTrue(same_runner_except_paths(legacy, current))
        self.assertTrue(same_runner_except_paths(current, current))
        # Changes to the controller, timing, or device binding are not metadata.
        for before, after in [('device="cuda:0"', 'device="cuda:1"'),
                              ('steps*12', 'steps*6'),
                              ('action = chunk.actions[0]', 'action = ctx.action()')]:
            self.assertIn(before, current)
            self.assertFalse(same_runner_except_paths(legacy, current.replace(before, after)))


if __name__ == "__main__":
    unittest.main()
