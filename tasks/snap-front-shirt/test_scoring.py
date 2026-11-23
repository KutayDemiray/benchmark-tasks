"""Adversarial diagnostic scorer checks; synthetic traces are not physics evidence."""

import copy
import json
from pathlib import Path
import unittest

from scoring import analyze

CFG = json.loads((Path(__file__).parent / 'parameters.json').read_text())


def sample(t, phase, height=.0066, load=0.):
    return {'time_s': t, 'phase': phase, 'male_pose_xyzw': [0, 0, .1 + height, 0, 0, 0, 1],
            'male_velocity': [0]*6, 'diagnostic_force_N': [0, 0, load], 'diagnostic_torque_Nm': [0]*3,
            'cloth': {'upper': {'finite': True, 'max_abs_edge_strain': .02, 'z_range_m': .01,
                                'max_rivet_error_m': .0001}}}


def valid_fixture():
    return [sample(0., 'press', load=-2.)] + [sample(.1+i*.0125, 'retention', load=1.) for i in range(40)] + [sample(.7, 'release', .014, 5.)]


class ScoreTests(unittest.TestCase):
    def test_isolated_success_never_full_task(self):
        result = analyze(valid_fixture(), CFG)
        self.assertTrue(result['retained_under_1N'])
        self.assertEqual(result['full_task_score'], 0.)
        self.assertFalse(result['full_task_success'])
        self.assertFalse(result['isolated_gate_pass'])

    def test_nearby_uninserted_and_wrong_lateral_pose(self):
        for axis, value in ((0, .004), (2, .1085)):
            rows = valid_fixture()
            for row in rows:
                row['male_pose_xyzw'][axis] = value
            self.assertFalse(analyze(rows, CFG)['retained_under_1N'])

    def test_held_by_press_cannot_count_as_retention(self):
        rows = valid_fixture()
        for row in rows:
            row['diagnostic_force_N'][2] = -2.
        self.assertFalse(analyze(rows, CFG)['retained_under_1N'])

    def test_single_frame_cannot_count_as_retention(self):
        self.assertFalse(analyze([sample(0., 'press', load=-2), sample(.1, 'retention', load=1)], CFG)['retained_under_1N'])

    def test_unseated_separation_is_not_release(self):
        self.assertFalse(analyze([sample(0., 'release', .02, 1.)], CFG)['released_under_limit'])

    def test_short_pull_and_inverted_approach(self):
        rows = valid_fixture()
        self.assertFalse(analyze(rows[:21], CFG)['retained_under_1N'])
        for row in rows:
            row['male_pose_xyzw'][3:] = [0., 1., 0., 0.]
        self.assertFalse(analyze(rows, CFG)['seated_during_press'])

    def test_invalid_and_damage_zero(self):
        for bad in ('nan', 'damage', 'rivet', 'quaternion', 'time', 'excess_force', 'excess_torque', 'missing', 'missing_rivet'):
            rows = copy.deepcopy(valid_fixture())
            if bad == 'nan': rows[2]['male_pose_xyzw'][2] = float('nan')
            if bad == 'damage': rows[2]['cloth']['upper']['max_abs_edge_strain'] = .5
            if bad == 'rivet': rows[2]['cloth']['upper']['max_rivet_error_m'] = .002
            if bad == 'quaternion': rows[2]['male_pose_xyzw'][6] = 0.
            if bad == 'time': rows[2]['time_s'] = rows[1]['time_s']
            if bad == 'excess_force': rows[2]['diagnostic_force_N'][2] = 100.
            if bad == 'excess_torque': rows[2]['diagnostic_torque_Nm'][0] = .1
            if bad == 'missing': del rows[2]['male_pose_xyzw']
            if bad == 'missing_rivet': del rows[2]['cloth']['upper']['max_rivet_error_m']
            result = analyze(rows, CFG)
            self.assertTrue(result['invalid_reasons'], bad)
            self.assertFalse(result['retained_under_1N'], bad)
            self.assertEqual(result['full_task_score'], 0.)


if __name__ == '__main__':
    unittest.main()
