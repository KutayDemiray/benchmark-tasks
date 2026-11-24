"""Keep scoring-only provenance compatibility from admitting other changes."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from episode_config import contract
from scoring_revision import digest, only_scorer_changed, verify_revision

ROOT = Path(__file__).resolve().parent


class RevisionTests(unittest.TestCase):
    def test_recorded_scores_and_revision_hashes_verify(self):
        evidence = verify_revision(ROOT)
        self.assertTrue(only_scorer_changed(evidence["episode_contract"], contract(ROOT)))

    def test_non_scoring_contract_changes_are_rejected(self):
        original = json.loads((ROOT/"evidence.json").read_text())["episode_contract"]
        current = contract(ROOT)
        for key, value in (("instruction", "Different instruction"), ("max_actions", 4000),
                           ("control_hz", 10), ("physics_hz", 120), ("seed", 1),
                           ("rgb_shape", [176, 320, 4]), ("cameras", ["privileged_cam"])):
            changed = copy.deepcopy(current)
            changed[key] = value
            with self.subTest(key=key):
                self.assertFalse(only_scorer_changed(original, changed))
        for name in current["source_hashes"]:
            if name == "scorer.py":
                continue
            changed = copy.deepcopy(current)
            changed["source_hashes"][name] = "different source"
            with self.subTest(file=name):
                self.assertFalse(only_scorer_changed(original, changed))
        self.assertFalse(only_scorer_changed(original, original))
        changed = copy.deepcopy(current)
        del changed["source_hashes"]["scorer.py"]
        self.assertFalse(only_scorer_changed(original, changed))

    def test_rehashed_reports_cannot_change_run_identity_or_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in ROOT.iterdir():
                if path.is_file():
                    shutil.copyfile(path, root/path.name)
            original = json.loads((ROOT/"evidence.json").read_text())
            item = original["scoring_revision"]["reports"]["model"]
            report = json.loads((ROOT/item["file"]).read_text())
            for key, value in (("trace_sha256", "wrong trace"), ("run_id", "different run"),
                               ("continuous_actions", 3599), ("scoring", {"score": 1.})):
                evidence, altered = copy.deepcopy(original), copy.deepcopy(report)
                altered[key] = value
                (root/item["file"]).write_text(json.dumps(altered))
                evidence["scoring_revision"]["reports"]["model"]["sha256"] = digest(root/item["file"])
                (root/"evidence.json").write_text(json.dumps(evidence))
                with self.subTest(key=key), self.assertRaises(AssertionError):
                    verify_revision(root)


if __name__ == "__main__":
    unittest.main()
