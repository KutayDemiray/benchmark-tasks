"""Geometric setup checks for a real-size shirt lying on a table."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from tabletop_scene import configure,layout
from tabletop_scoring import task_score,seating_matrix
from robot_scoring import snap_seated


class TabletopChecks(unittest.TestCase):
    def test_rotation_preserves_full_shirt_metric_and_clears_table(self):
        cfg=json.loads((Path(__file__).parent/'parameters.json').read_text());configure(cfg)
        data,rotation=layout(cfg)
        self.assertTrue(np.allclose(rotation.T@rotation,np.eye(3)))
        lengths=np.linalg.norm(data['initial'][data['edges'][:,0]]-data['initial'][data['edges'][:,1]],axis=1)
        self.assertLess(float(np.max(np.abs(lengths/data['rest_lengths']-1))),1e-10)
        self.assertAlmostEqual(data['initial'][:,2].min(),cfg['tabletop']['table_top_m']+.0005)
        self.assertEqual(data['audit']['boundary_loops'],3)
        self.assertEqual(data['audit']['connected_components'],1)
        self.assertEqual(len(data['rivets']),14)
        self.assertEqual(cfg['tabletop']['required_snap_indices'],[0,1,2])
        self.assertGreater(data['audit']['fabric_mass_kg'],.15)


class TabletopScoreChecks(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((Path(__file__).parent/'parameters.json').read_text());configure(self.cfg)
        # Compact synthetic traces test predicates, not a simulated success.
        self.cfg['simulation']['physics_hz']=10
        initial={}
        for i in range(7):
            initial[f'socket{i}']=[.28+.08*i,.12,.752,0.,0.,0.,1.]
            initial[f'male{i}']=[.28+.08*i,.12,.772,0.,0.,0.,1.]
        self.meta=dict(diagnostic_world_mounts=False,grasp_attachments=False,hanger_attachments=False,
            initial_hardware=copy.deepcopy(initial),cloth={'attachments':[{} for _ in range(14)]})
        self.physics=[]
        for k in range(53):
            hardware=copy.deepcopy(initial)
            if k:
                for i in range(3):hardware[f'male{i}'][2]=.7588
            self.physics.append(dict(physics_step=k,hardware=hardware,strain=.01,rivet_error_m=.0001,
                table_penetration_m=0.,table_coverage_fraction=1.,finite=True,
                mean_cloth_position_m=[.55,.12,.754],snap_robot_contact_norm_N={n:0. for n in hardware}))
        self.rows=[dict(step=i,time_s=(i+1)/10,robot_action=[0.]*14,physics=self.physics[i+1],
            robot_clearance_m=dict(fabric_m=.10,hardware_m=.12),control={'phase':'arbitrary'}) for i in range(52)]

    def score(self):
        return task_score(self.rows,self.physics,self.cfg,self.meta,'time_limit')

    def test_three_retained_pairs_receive_full_credit_without_controller_labels(self):
        result=self.score()
        self.assertEqual(result['task_score'],1.)
        self.assertTrue(result['full_task_success'])
        self.assertFalse(result['original_hanger_task_success'])

    def test_one_retained_pair_is_only_one_third_credit(self):
        for r in self.physics:
            for i in (1,2):r['hardware'][f'male{i}'][2]=.772
        self.assertEqual(self.score()['task_score'],1/3)

    def test_proximity_above_detents_is_not_fastening(self):
        for r in self.physics[1:]:
            for i in range(3):r['hardware'][f'male{i}'][2]=.7615
        self.assertEqual(self.score()['task_score'],0.)

    def test_cap_forced_through_socket_rim_is_not_fastening(self):
        for r in self.physics[1:]:
            for i in range(3):r['hardware'][f'male{i}'][2]=.755
        self.assertEqual(self.score()['task_score'],0.)

    def test_one_cap_through_rim_invalidates_otherwise_valid_partial_credit(self):
        self.physics[-1]['hardware']['male0'][2]=.755
        result=self.score()
        self.assertEqual(result['task_score'],0.)
        self.assertIn('cap_through_socket_rim',result['invalid_reasons'])

    def test_any_robot_support_prevents_credit(self):
        for r in self.rows:r['robot_clearance_m']['fabric_m']=.0001
        self.assertEqual(self.score()['task_score'],0.)

    def test_wrong_pairing_is_possible_in_state_but_cannot_earn_credit(self):
        for r in self.physics[1:]:r['hardware']['male0'][:3]=[.36,.12,.758]
        self.assertEqual(self.score()['task_score'],0.)

    def test_initial_closure_is_rejected(self):
        self.physics[0]['hardware']['male0'][2]=.758
        self.assertIn('initially_closed_snap',self.score()['invalid_reasons'])

    def test_damage_cannot_be_repaired_in_scoring_by_later_rest(self):
        self.physics[2]['strain']=.16
        self.assertIn('cloth_strain_limit',self.score()['invalid_reasons'])

    def test_nonfinite_unrequired_hardware_invalidates_episode(self):
        self.physics[2]['hardware']['male6'][0]=float('nan')
        self.assertIn('nonfinite_hardware',self.score()['invalid_reasons'])

    def test_nonfinite_cloth_position_is_not_stability(self):
        self.rows[-1]['physics']['mean_cloth_position_m'][0]=float('nan')
        self.assertEqual(self.score()['task_score'],0.)

    def test_short_retention_cannot_earn_credit(self):
        for r in self.physics[:35]:
            for i in range(3):r['hardware'][f'male{i}'][2]=.772
        self.assertEqual(self.score()['task_score'],0.)

    def test_hidden_grasp_attachment_invalidates_episode(self):
        self.meta['grasp_attachments']=True
        self.assertIn('hidden_attachment',self.score()['invalid_reasons'])

    def test_nonfinite_secondary_contact_cannot_hide_behind_maximum(self):
        self.physics[5]['snap_robot_contact_norm_N']['socket6']=float('nan')
        self.assertIn('invalid_contact_measurement',self.score()['invalid_reasons'])

    def test_control_and_native_state_streams_must_correspond(self):
        self.rows[-1]['physics']=copy.deepcopy(self.physics[-1])
        self.rows[-1]['physics']['mean_cloth_position_m'][0]+=.1
        self.assertIn('control_physics_mismatch',self.score()['invalid_reasons'])

    def test_impossible_table_coverage_is_invalid(self):
        self.physics[4]['table_coverage_fraction']=1.1
        self.assertIn('invalid_table_coverage',self.score()['invalid_reasons'])

    def test_batched_pairing_matches_scalar_check_including_wrong_pairs(self):
        rng=np.random.default_rng(120)
        for r in self.physics[1:]:
            for i in range(7):r['hardware'][f'male{i}'][0]+=rng.normal(0,.001)
        batch=seating_matrix(self.physics,7,self.cfg)
        for k in (0,1,12,52):
            for i in range(7):
                for j in range(7):
                    self.assertEqual(bool(batch[k,i,j]),snap_seated(self.physics[k]['hardware'][f'male{i}'],
                        self.physics[k]['hardware'][f'socket{j}'],self.cfg))

    def test_pairing_is_invariant_to_a_rigid_change_of_world_frame(self):
        from scipy.spatial.transform import Rotation
        before=seating_matrix(self.physics,7,self.cfg)
        frame=Rotation.from_rotvec([.7,-.4,.3]);offset=np.array([1.,-2.,.4])
        for row in self.physics:
            for pose in row['hardware'].values():
                pose[:3]=(frame.apply(pose[:3])+offset).tolist()
                pose[3:]=(frame*Rotation.from_quat(pose[3:])).as_quat().tolist()
        after=seating_matrix(self.physics,7,self.cfg)
        self.assertTrue(np.array_equal(before,after))


if __name__=='__main__':
    unittest.main()
