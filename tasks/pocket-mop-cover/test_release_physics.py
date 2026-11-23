"""Reject physical-surface removal and accidental diagnostic use in full episodes."""

from copy import deepcopy
import itertools
from pathlib import Path
import unittest

import numpy as np

from release_physics import install, redundant_clamps


class ReleasePhysicsTests(unittest.TestCase):
    def geometry(self):
        corners=np.array(list(itertools.product((-1.,1.),repeat=3)))
        shapes=[]
        for body in ('Left','Right'):
            for name,points in [('End',corners),('ClosedStripClamp',corners*[.5,.5,.1]+[0,0,.9])]:
                shapes.append(dict(body=body,name=name,points=points.tolist(),enabled=True))
        return dict(colliders=shapes)

    def test_coplanar_nested_collision_volumes_are_redundant(self):
        proof=redundant_clamps(self.geometry())
        self.assertEqual(len(proof),2)
        self.assertTrue(all(item['convex_union_unchanged'] for item in proof))

    def test_exposed_material_cannot_be_removed(self):
        geometry=deepcopy(self.geometry())
        geometry['colliders'][1]['points'][0][2]=1.001
        with self.assertRaisesRegex(AssertionError,'solid boundary'):
            redundant_clamps(geometry)

    def test_numerical_controls_cannot_enter_ordinary_episode(self):
        for key,value in [('release_probe_depenetration','frame'),
                          ('release_probe_deduplicate_colliders',True),
                          ('release_probe_velocity_iterations',0),
                          ('release_probe_velocity_iterations',1),
                          ('release_probe_velocity_iterations',32),
                          ('release_probe_no_speculative_ccd',True)]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                install({key:value},Path('/unused'))

    def test_solver_order_override_rejects_an_ordinary_episode(self):
        from release_physics import configure_scene
        with self.assertRaises(AssertionError):
            configure_scene({'release_probe_contact_last': True}, None)
