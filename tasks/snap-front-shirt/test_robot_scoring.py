"""Adversarial checks for robot diagnostic evidence."""

import copy
import unittest

from robot_scoring import analyze_grasp


class RobotScoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'gate': {'max_edge_strain': .15, 'max_attachment_error_m': .001},
                    'simulation': {'physics_hz': 10}}
        self.topology = {'diagnostic_world_mounts': False, 'grasp_attachments': False,
                         'initial_hardware': {'male': [0, 0, .95, 0, 0, 0, 1]}}
        self.control = []
        self.physics = []
        for i in range(131):
            t = i/10
            phase = 'hold' if 8 <= t < 10 else 'retreat' if t > 11 else 'approach'
            grip = 0. if phase == 'hold' else 1.
            z = 1.01 if phase == 'hold' else .95
            p = {'physics_step': i, 'hardware': {'male': [0, 0, z, 0, 0, 0, 1]},
                 'cloth': {'coupon': {'strain': .03, 'rivet_error_m': .0001, 'finite': True}}}
            self.physics.append(p)
            if i:
                self.control.append({'time_s': t, 'control': {'phase': phase},
                                     'robot_action': [0]*6+[grip]+[0]*6+[1.],
                                     'joint_positions': [0]*6+[grip]+[0]*6+[1.], 'physics': p})

    def score(self):
        return analyze_grasp(self.control, self.physics, self.cfg, self.topology)

    def test_valid_lift_release_never_full_task(self):
        result = self.score()
        self.assertTrue(result['lifted_15mm'] and result['released_after_opening'])
        self.assertEqual(result['full_task_score'], 0.)
        self.assertFalse(result['isolated_gate_pass'])

    def test_physics_spike_between_actions_invalidates(self):
        self.physics[30]['cloth']['coupon']['strain'] = .2
        self.assertFalse(self.score()['lifted_15mm'])

    def test_drop_without_previous_lift_is_not_release(self):
        for row in self.physics:
            row['hardware']['male'][2] = .95
        self.assertFalse(self.score()['released_after_opening'])

    def test_missing_substeps_invalidates(self):
        self.physics.pop(40)
        self.assertIn('missing_physics_steps', self.score()['invalid_reasons'])

    def test_attachment_and_bad_actions_invalid(self):
        self.topology['grasp_attachments'] = True
        self.assertFalse(self.score()['cloth_integrity'])
        self.topology['grasp_attachments'] = False
        self.control[30]['robot_action'][0] = float('nan')
        self.assertIn('invalid_robot_action', self.score()['invalid_reasons'])

    def test_transient_lift_not_retention(self):
        for row in self.control:
            if row['time_s'] < 9.8:
                row['control']['phase'] = 'approach'
        self.assertFalse(self.score()['lifted_15mm'])


class RobotPressScoreTests(unittest.TestCase):
    def setUp(self):
        import json
        from pathlib import Path
        self.cfg = json.loads((Path(__file__).parent/'parameters.json').read_text())
        self.cfg['simulation']['physics_hz'] = 10
        self.topology = {'diagnostic_world_mounts': False, 'grasp_attachments': False,
                         'snap_pair_present': True,
                         'initial_hardware': {'male': [.3,.12,.915,0,0,0,1],
                                              'socket': [.3,.12,.901,0,0,0,1]}}
        self.control, self.physics = [], []
        for i in range(141):
            t = i/10
            phase = 'press' if 5 <= t < 9 else 'released_observation' if t >= 11 else 'approach'
            z = .9078 if t >= 6 else .915
            p = {'physics_step': i,
                 'hardware': {'male': [.3,.12,z,0,0,0,1], 'socket': [.3,.12,.901,0,0,0,1]},
                 'cloth': {'upper': {'strain': .03, 'rivet_error_m': .0001, 'finite': True}},
                 'male_robot_normal_forces_N': [[0,0,-2 if phase=='press' else 0], [0,0,0]]}
            self.physics.append(p)
            if i:
                self.control.append({'time_s': t, 'control': {'phase': phase},
                                     'robot_action': [0]*14, 'measured_tcp_m': [.3,.12,.95], 'physics': p})

    def score(self):
        from robot_scoring import analyze_press
        return analyze_press(self.control, self.physics, self.cfg, self.topology)

    def test_contact_seating_then_release_never_full_gate(self):
        r = self.score()
        self.assertTrue(r['seated_after_robot_press'])
        self.assertFalse(r['isolated_gate_pass'])
        self.assertEqual(r['full_task_score'], 0.)

    def test_controller_phase_names_do_not_change_outcome(self):
        for row in self.control:
            row['control']['phase'] = 'arbitrary_controller_label'
        self.assertTrue(self.score()['seated_after_robot_press'])

    def test_gravity_seating_without_prior_robot_contact_fails(self):
        for row in self.physics:
            if row['physics_step']<50:
                row['male_robot_normal_forces_N'] = [[0,0,0],[0,0,0]]
            if row['physics_step']>=30:
                row['hardware']['male'][2] = .9078
        r = self.score()
        self.assertTrue(r['uncommanded_seating_detected'])
        self.assertFalse(r['seated_after_robot_press'])

    def test_no_force_cannot_prove_robot_press(self):
        for p in self.physics:
            p['male_robot_normal_forces_N'] = [[0,0,0],[0,0,0]]
        self.assertFalse(self.score()['seated_after_robot_press'])

    def test_preclosed_and_held_by_hand_fail(self):
        for r in self.control:
            r['measured_tcp_m'][2] = .92
        self.assertFalse(self.score()['seated_after_robot_press'])
        self.topology['initial_hardware']['male'][2] = .9078
        self.assertIn('initially_seated_pair',self.score()['invalid_reasons'])

    def test_force_and_substep_strain_spikes_fail(self):
        self.physics[32]['male_robot_normal_forces_N'][0][2] = -30
        self.assertFalse(self.score()['seated_after_robot_press'])
        self.physics[32]['male_robot_normal_forces_N'][0][2] = 0
        self.physics[32]['cloth']['upper']['strain'] = .2
        self.assertFalse(self.score()['seated_after_robot_press'])

    def test_changed_control_copy_and_missing_physics_fail(self):
        self.control[65]['physics'] = copy.deepcopy(self.control[65]['physics'])
        self.control[65]['physics']['hardware']['male'][2] += .001
        self.assertIn('mismatched_control_and_physics',self.score()['invalid_reasons'])
        self.physics.pop(30)
        self.assertIn('missing_physics_steps',self.score()['invalid_reasons'])

    def test_seating_uses_moving_frame_not_world_height(self):
        from robot_scoring import snap_seated
        # A 90-degree rotation about world Y maps insertion direction to world X.
        q = [0,2**-.5,0,2**-.5]
        self.assertTrue(snap_seated([1.0068,2,3,*q],[1,2,3,*q],self.cfg))
        self.assertFalse(snap_seated([1.009,2,3,*q],[1,2,3,*q],self.cfg))
        self.assertFalse(snap_seated([1.0068,2.004,3,*q],[1,2,3,*q],self.cfg))
        self.assertFalse(snap_seated([1.0068,2,3,0,0,0,1],[1,2,3,*q],self.cfg))


class RobotRetentionTests(unittest.TestCase):
    def setUp(self):
        base=RobotPressScoreTests()
        base.setUp()
        self.cfg,self.topology=base.cfg,base.topology
        self.topology.update(support_top_m=.9,support_front_x_m=.2)
        self.physics=base.physics
        for i in range(len(self.physics),341):
            row=copy.deepcopy(self.physics[-1]);row['physics_step']=i;self.physics.append(row)
        self.control=[]
        for i,row in enumerate(self.physics):
            t=i/10
            held=25<=t<27
            if held:
                row['hardware']['male'][2]=.9578
                row['hardware']['socket'][2]=.951
            row['socket_robot_normal_forces_N']=[[0,0,0],[0,0,0]]
            phase='hold' if held else 'change_to_cloth_grasp' if 14<=t<17 else 'retreat'
            if i:
                self.control.append({'time_s':t,'control':{'phase':phase},'robot_action':[0]*14,
                    'joint_positions':[0]*6+[0 if held else 1]+[0]*6+[1],
                    'physics':row,'measured_tcp_m':[.3,.12,1.05],
                    'lower_robot_clearance_m':{'fabric_m':.008,'socket_m':.02},
                    'upper_hardware_clearance_m':.015, 'upper_cloth_robot_clearance_m':.01})

    def score(self):
        from robot_scoring import analyze_press_lift
        return analyze_press_lift(self.control,self.physics,self.cfg,self.topology)

    def test_lower_lift_and_release_are_only_intermediate(self):
        r=self.score()
        self.assertTrue(r['lifted_pair_by_upper_fabric'] and r['released_pair_on_support'])
        self.assertEqual(r['full_task_score'],0.)
        self.assertFalse(r['isolated_gate_pass'])

    def test_table_support_is_not_pair_lift(self):
        for row in self.physics[250:270]:
            row['hardware']['male'][2]=.9078;row['hardware']['socket'][2]=.901
        self.assertFalse(self.score()['lifted_pair_by_upper_fabric'])

    def test_second_hand_or_lower_cloth_grasp_cannot_prove_snap_retention(self):
        self.control[254]['lower_robot_clearance_m']['fabric_m']=0
        self.assertFalse(self.score()['lifted_pair_by_upper_fabric'])
        self.control[254]['lower_robot_clearance_m']['fabric_m']=.008
        self.physics[254]['socket_robot_normal_forces_N'][1][2]=1
        self.assertFalse(self.score()['lifted_pair_by_upper_fabric'])

    def test_open_fingers_with_cloth_still_on_robot_is_not_release(self):
        self.control[-4]['upper_cloth_robot_clearance_m'] = 0.
        result = self.score()
        self.assertTrue(result['lifted_pair_by_upper_fabric'])
        self.assertFalse(result['released_pair_on_support'])

    def test_release_beyond_narrow_support_edge_fails(self):
        self.topology['support_width_m'] = .06
        for row in self.physics[-20:]:
            for pose in row['hardware'].values():
                pose[1] = .17
        result = self.score()
        self.assertTrue(result['lifted_pair_by_upper_fabric'])
        self.assertFalse(result['released_pair_on_support'])

    def test_holding_upper_metal_is_not_fabric_grasp_evidence(self):
        self.control[254]['upper_hardware_clearance_m'] = 0.
        result = self.score()
        self.assertFalse(result['lifted_pair_by_upper_fabric'])
        self.assertIn('upper_hardware_directly_grasped',result['invalid_reasons'])

    def test_opened_snap_during_hold_is_not_retention(self):
        self.physics[254]['hardware']['male'][2]+=.005
        self.assertFalse(self.score()['lifted_pair_by_upper_fabric'])

    def test_clearance_agrees_before_and_after_json_roundtrip(self):
        import json
        import numpy as np
        from robot_scoring import lower_clearance
        nodes=np.array([[.32,.14,.91],[.322,.14,.91],[.32,.142,.91]],dtype=np.float32)
        planes=np.array([[1,0,0,.005],[-1,0,0,.005],[0,1,0,.005],
                         [0,-1,0,.005],[0,0,1,.005],[0,0,-1,.005]])
        topology={'cloth':{'lower':{'edges':[[0,1],[1,2],[2,0]]}},
                  'robot_collision_planes':{k:[planes] for k in ('gripper','tip_left','tip_right')}}
        row={'physics':{'hardware':{'socket':[.32,.14,.90,0,0,0,1]}}}
        for side in ('left','right'):
            row[side+'_link_names']=['gripper']
            row[side+'_link_poses_xyzw']=np.array([[.27,.12,.895,0,.70710678,0,.70710678]],dtype=np.float32)
        expected=lower_clearance(row,nodes,topology)
        decoded=json.loads(json.dumps(row,default=lambda x:x.tolist()))
        self.assertEqual(expected,lower_clearance(decoded,nodes,topology))


if __name__ == '__main__':
    unittest.main()
