"""Check single-substep compression detection and bounded event history."""

import unittest
import numpy as np
from strain_witness import StrainWitness


class WitnessChecks(unittest.TestCase):
    def test_compression_spike_is_retained_after_relaxation(self):
        witness = StrainWitness([[0, 1]], [.002], [[0, 1, 2]], history_frames=2)
        for step, strain in enumerate([0., .01, -.16, .02, .01, 0.]):
            nodes = np.array([[0., 0., 0.], [.002*(1+strain), 0., 0.], [0., .01, 0.]])
            witness.record(dict(physics_step=step, reported_strain=abs(strain),
                                reported_edge_index=0, cloth_positions_m=nodes))
        result = witness.audit()
        self.assertTrue(result['cpu_agreement'])
        self.assertEqual(result['classification'], 'compression')
        self.assertEqual(result['physics_step'], 2)
        self.assertEqual(result['history_physics_steps'], [0, 1, 2, 3, 4])
        self.assertAlmostEqual(result['signed_strain'], -.16)

    def test_new_global_maximum_replaces_the_previous_history(self):
        witness = StrainWitness([[0, 1]], [1.], [[0, 1, 2]], history_frames=1)
        for step, strain in enumerate([0., .05, .01, .1, 0.]):
            nodes = np.array([[0., 0., 0.], [1+strain, 0., 0.], [0., 1., 0.]])
            witness.record(dict(physics_step=step, reported_strain=strain,
                                reported_edge_index=0, cloth_positions_m=nodes))
        result = witness.audit()
        self.assertEqual(result['history_physics_steps'], [2, 3, 4])
        self.assertEqual(result['classification'], 'extension')
        self.assertTrue(result['cpu_agreement'])

    def test_incorrect_native_summary_is_detected(self):
        witness = StrainWitness([[0, 1]], [1.], [[0, 1, 2]])
        witness.record(dict(physics_step=0, reported_strain=.2, reported_edge_index=0,
                            cloth_positions_m=np.array([[0., 0., 0.], [1.1, 0., 0.], [0., 1., 0.]])))
        self.assertFalse(witness.audit()['cpu_agreement'])
