"""Guard revised score routing and preservation of verified invalid outcomes."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation_contract import assessment_paths
from score_handle_held import score


class RevisedEvaluationTests(unittest.TestCase):
    def fixture(self, root, policy='dream'):
        request = dict(robot_cover=True, completion_mode='handle-held', policy=policy, physics_hz=480)
        (root/'request.json').write_text(json.dumps(request))
        (root/'result.json').write_text(json.dumps(dict(status='completed', control_steps=1)))
        witness = dict(reason='material_strain', scope='synthetic CPU control fixture')
        baseline = dict(score=0., success=False, invalid_reasons=['material_strain'], native_counterexample=witness)
        (root/'benchmark-result.json').write_text(json.dumps(baseline))
        return witness

    def test_each_rubric_uses_its_own_assessment(self):
        self.assertEqual(assessment_paths({})[1], 'benchmark-result.json')
        self.assertEqual(assessment_paths({'completion_mode':'handle-held'}),
            ('handle-held-v2', 'handle-held-benchmark-result.json', 'handle-held-outcomes.jsonl'))
        with self.assertRaises(AssertionError):
            assessment_paths({'completion_mode':'unknown'})

    def test_verified_native_zero_is_preserved_without_rewriting_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);witness=self.fixture(root)
            original=(root/'benchmark-result.json').read_bytes()
            with patch('invalid_witness.strain_witness', return_value=witness):
                result=score(root)
            self.assertEqual(result['score'], 0.)
            self.assertFalse(result['success'])
            self.assertEqual(result['rubric_revision'], 'handle-held-v2')
            self.assertEqual(result['native_counterexample'], witness)
            self.assertEqual(result['termination_reason'], 'invalid')
            self.assertEqual((root/'benchmark-result.json').read_bytes(), original)
            self.assertEqual(result['original_assessment_sha256'], hashlib.sha256(original).hexdigest())

    def test_unverified_native_zero_and_mock_episode_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with patch('invalid_witness.strain_witness', return_value=None):
                with self.assertRaises(AssertionError):score(root)
            self.fixture(root, policy='interface-check')
            with self.assertRaisesRegex(AssertionError, 'Offline interface'):
                score(root)


if __name__ == '__main__':
    unittest.main()
