"""Packaging must report genuine model activity without inventing a score."""

import json
from pathlib import Path
import tempfile
import unittest

from package_revision import collect_model_activity, model_activity


class ModelActivityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.run=Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self,name,value):
        (self.run/name).write_text(json.dumps(value))

    def live(self):
        self.write('manifest.json',{'command':['python','/task/dream_live_runner.py']})

    def test_offline_fake_session_is_not_a_hosted_attempt(self):
        self.write('evaluation.json',{'mode':'fake','model_call':False})
        self.assertIsNone(model_activity(self.run))

    def test_zero_score_live_attempt_is_preserved(self):
        self.live()
        (self.run/'inference.jsonl').write_text(json.dumps({'event':'started'})+'\n'+json.dumps({'event':'completed'})+'\n')
        self.write('model-evaluation.json',dict(mode='dream_live',model='dreamlabs/dream-1-beta',
            hosted_requests_started=1,hosted_requests_completed=1,error=None,
            result={'task_score':0.,'control_steps':50,'termination_reason':'invalid_simulation_state'}))
        record=model_activity(self.run)
        self.assertTrue(record['model_attempt'])
        self.assertEqual(record['task_score'],0.)
        self.assertEqual(record['executed_control_steps'],50)
        self.assertFalse(record['independently_audited'])

    def test_pre_request_failure_is_not_an_attempt_or_zero_score(self):
        self.live()
        self.write('failure.json',{'error':'missing access','hosted_requests_started':0})
        record=model_activity(self.run)
        self.assertFalse(record['model_attempt'])
        self.assertIsNone(record['task_score'])
        self.assertEqual(record['error']['error'],'missing access')

    def test_failed_request_still_counts_as_an_attempt(self):
        self.live()
        (self.run/'inference.jsonl').write_text(json.dumps({'event':'started'})+'\n'+json.dumps({'event':'error'})+'\n')
        self.write('failure.json',{'error':'request failed','hosted_requests_started':1})
        record=model_activity(self.run)
        self.assertTrue(record['model_attempt'])
        self.assertEqual(record['inference_requests_completed'],0)
        self.assertIsNone(record['task_score'])

    def test_nested_duplicate_records_do_not_duplicate_the_attempt(self):
        activity={'path':'same-run','model_attempt':True}
        records=[{'model_activity':activity,'children':[{'model_activity':activity}]}]
        self.assertEqual(collect_model_activity(records),[activity])


if __name__=='__main__':unittest.main()
