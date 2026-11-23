"""Check bounded passive-screen decisions without launching native workers."""

import unittest
from stability_round import CANDIDATES, valid_passive


class StabilityChecks(unittest.TestCase):
    def test_pass_requires_complete_horizon_and_no_invalid_outcomes(self):
        record = dict(diagnostic_pass=True,invalid_reasons=[],termination_reason='diagnostic_completed',control_steps=3000)
        self.assertTrue(valid_passive(record,300))
        self.assertFalse(valid_passive({**record,'control_steps':2999},300))
        self.assertFalse(valid_passive({**record,'invalid_reasons':['cloth_strain_limit']},300))
        self.assertFalse(valid_passive({**record,'termination_reason':'invalid_fabric_state'},300))
        self.assertFalse(valid_passive({},300))

    def test_candidate_order_and_bound_are_explicit(self):
        self.assertEqual([(x['iterations'],x['pitch']) for x in CANDIDATES],[(64,.01),(128,.01),(64,.005)])


class RefinementChecks(unittest.TestCase):
    def test_refinement_keeps_hardware_attachment_sampling_and_geometry(self):
        import json
        from pathlib import Path
        import numpy as np
        from tabletop_scene import configure, layout
        configurations = []
        for pitch in (.01,.005):
            cfg = json.loads((Path(__file__).parent/'parameters.json').read_text())
            configure(cfg)
            cfg['garment']['body_mesh_pitch_m'] = pitch
            data, _ = layout(cfg)
            configurations.append(data)
        coarse, fine = configurations
        self.assertGreater(len(fine['initial']),len(coarse['initial']))
        self.assertEqual(coarse['audit']['rivet_patch_vertices'],fine['audit']['rivet_patch_vertices'])
        self.assertLess(abs(fine['audit']['fabric_mass_kg']/coarse['audit']['fabric_mass_kg']-1),.003)
        for first, second in zip(coarse['rivets'],fine['rivets'],strict=True):
            np.testing.assert_allclose(first['center'],second['center'],atol=1e-12)
        self.assertAlmostEqual(coarse['initial'][:,2].min(),fine['initial'][:,2].min())
