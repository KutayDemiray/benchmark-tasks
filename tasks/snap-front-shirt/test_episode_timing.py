"""Check equal physical retention at both supported action frequencies."""

import copy
import unittest

from episode_timing import control_hz,physics_stride,max_actions,retention_samples
from tabletop_scoring import task_score
import test_tabletop


class TimingChecks(unittest.TestCase):
    def test_rates_preserve_physical_time(self):
        for hz,stride,actions,samples in ((10,96,3000,31),(20,48,6000,61)):
            cfg = dict(completion=dict(control_hz=hz,run_limit_seconds=300),
                       simulation=dict(physics_hz=960),tabletop=dict(released_observation_seconds=3))
            self.assertEqual(control_hz(cfg),hz)
            self.assertEqual(physics_stride(cfg),stride)
            self.assertEqual(max_actions(cfg),actions)
            self.assertEqual(retention_samples(cfg),samples)
            self.assertEqual((samples-1)/hz,3)

    def test_twenty_hz_cannot_claim_three_seconds_from_thirty_one_frames(self):
        fixture = test_tabletop.TabletopScoreChecks()
        fixture.setUp()
        cfg = copy.deepcopy(fixture.cfg)
        cfg['completion']['control_hz'] = 20
        cfg['simulation']['physics_hz'] = 20
        rows = copy.deepcopy(fixture.rows[:31])
        physics = copy.deepcopy(fixture.physics[:32])
        for i,row in enumerate(rows):
            row['time_s'] = (i+1)/20
            row['physics'] = physics[i+1]
        result = task_score(rows,physics,cfg,fixture.meta,'time_limit')
        self.assertFalse(result['invalid_reasons'])
        self.assertEqual(result['task_score'],0.)
        for i in range(31,62):
            state = copy.deepcopy(physics[-1]);state['physics_step']=i+1;physics.append(state)
            row = copy.deepcopy(rows[-1]);row.update(step=i,time_s=(i+1)/20,physics=state);rows.append(row)
        self.assertEqual(task_score(rows,physics,cfg,fixture.meta,'time_limit')['task_score'],1.)
