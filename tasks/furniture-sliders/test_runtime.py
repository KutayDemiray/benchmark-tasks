"""Fail closed before importing CUDA when invoked without the right reservation."""
import os
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest import TestCase,main
from unittest.mock import patch
from scheduled_runtime import reservation
from run import compatible_gate,diagnostic_reset
from config import S


class RuntimeTests(TestCase):
    def test_full_rejects_initialized_reset(self):
        with self.assertRaisesRegex(ValueError,'restricted'):
            diagnostic_reset('full',Path('arbitrary-reset.json'))
        self.assertIsNone(diagnostic_reset('full',None))

    def test_rear_reset_requires_all_pads(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'reset.json'
            p.write_text(json.dumps({'measurement':{'furniture':[0,0,0,0,0,0,1],
                                                   'pads':{'slider_0':[0,0,0,0,0,0,1]}}}))
            with self.assertRaisesRegex(ValueError,'four pad'):diagnostic_reset('rear-debug',p)

    def test_no_reservation(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'gpu-scheduler'): reservation()

    def test_other_task_cannot_be_reused(self):
        with patch.dict(os.environ,{'ROBOCURVE_TASK_ID':'pocket-mop-cover'},clear=True):
            with self.assertRaisesRegex(RuntimeError,'furniture-sliders'): reservation()

    def test_gate_rejects_contact_changes_but_allows_budget_review(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'task-bundle').mkdir()
            raw=json.dumps(S).encode()
            (p/'task-bundle'/'spec.json').write_bytes(raw)
            (p/'spec-used.json').write_bytes(raw)
            gate={'one_pad_gate_pass':True,'device':{'verified':True},
                  'spec_sha256':hashlib.sha256(raw).hexdigest()}
            result=p/'result.json';result.write_text(json.dumps(gate))
            changed=copy.deepcopy(S);changed['control']['max_steps']+=2000
            self.assertTrue(compatible_gate(result,changed))
            changed['materials']['foam']['static']+=.1
            self.assertFalse(compatible_gate(result,changed))
            gate['one_pad_gate_pass']=False;result.write_text(json.dumps(gate))
            self.assertFalse(compatible_gate(result,S))


if __name__=='__main__': main()
