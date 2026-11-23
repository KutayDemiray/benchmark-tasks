"""A negative certificate must use native geometry, never a logged hint alone."""

from task_paths import OUTPUT_ROOT

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from invalid_witness import strain_witness


class InvalidWitnessTests(unittest.TestCase):
    def test_logged_hint_requires_actual_native_violation(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as tmp:
            run=Path(tmp)
            (run/'physics-chunks').mkdir()
            (run/'request.json').write_text(json.dumps(dict(physics_hz=480)))
            (run/'manifest.json').write_text('{}')
            (run/'steps.jsonl').write_text(json.dumps(dict(max_abs_edge_strain=.4))+'\n')
            rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
            np.savez(run/'geometry.npz',rest=rest,faces=[[0,1,2]])
            points=np.repeat(rest[None],24,axis=0)
            np.savez(run/'physics-chunks/00000.npz',positions=points)
            self.assertIsNone(strain_witness(run))
            points[7,1,0]=1.25
            np.savez(run/'physics-chunks/00000.npz',positions=points)
            witness=strain_witness(run)
            self.assertEqual(witness['physics_step'],7)
            self.assertAlmostEqual(witness['absolute_strain'],.25)
            self.assertEqual(witness['threshold'],.2)
            self.assertIn('physics-chunks/00000.npz',witness['source_sha256'])


if __name__=='__main__':
    unittest.main()
