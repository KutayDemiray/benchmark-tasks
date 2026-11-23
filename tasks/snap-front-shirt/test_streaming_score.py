"""Compare streaming scores against independently assembled offline prefixes."""

import copy
import unittest
import test_tabletop
from streaming_score import StreamingScore
from tabletop_scoring import task_score


class StreamingChecks(unittest.TestCase):
    def compare(self, mutate=None):
        fixture=test_tabletop.TabletopScoreChecks();fixture.setUp()
        if mutate is not None:
            mutate(fixture)
        scorer=StreamingScore(fixture.cfg,fixture.meta)
        scorer.observe_physics(fixture.physics[0])
        for i,row in enumerate(fixture.rows):
            scorer.observe_physics(fixture.physics[i+1])
            online=scorer.observe_control(row)
            offline=task_score(fixture.rows[:i+1],fixture.physics[:i+2],fixture.cfg,fixture.meta,'running')
            self.assertEqual(online['task_score'],offline['task_score'],i)
            self.assertEqual(online['retained_pairs'],offline['retained_pairs'],i)
            self.assertEqual(bool(online['invalid_reasons']),bool(offline['invalid_reasons']),i)
        return scorer.result()

    def test_full_closure_and_retention_prefixes(self):
        self.assertEqual(self.compare()['task_score'],1.)

    def test_one_closed_pair_remains_partial_credit(self):
        def modify(f):
            for row in f.physics:
                for i in (1,2):row['hardware'][f'male{i}'][2]=.772
        self.assertEqual(self.compare(modify)['task_score'],1/3)

    def test_late_damage_permanently_invalidates_score(self):
        def modify(f):f.physics[45]['strain']=.16
        self.assertEqual(self.compare(modify)['task_score'],0.)

    def test_late_rim_penetration_invalidates_other_closures(self):
        def modify(f):f.physics[45]['hardware']['male6'][2]=.755
        self.assertEqual(self.compare(modify)['task_score'],0.)

    def test_a_wrong_pair_during_release_prevents_credit(self):
        def modify(f):
            for row in f.physics[1:]:row['hardware']['male0'][:3]=[.36,.12,.758]
        self.assertEqual(self.compare(modify)['task_score'],0.)

    def test_robot_support_prevents_credit(self):
        def modify(f):
            for row in f.rows:row['robot_clearance_m']['fabric_m']=.001
        self.assertEqual(self.compare(modify)['task_score'],0.)


    def test_twenty_hz_with_multiple_native_states_per_action(self):
        fixture=test_tabletop.TabletopScoreChecks();fixture.setUp()
        cfg=copy.deepcopy(fixture.cfg)
        cfg['completion']['control_hz']=20
        cfg['simulation']['physics_hz']=40
        scorer=StreamingScore(cfg,fixture.meta)
        physics=[copy.deepcopy(fixture.physics[0])]
        scorer.observe_physics(physics[0])
        rows=[]
        for i in range(70):
            for _ in range(2):
                state=copy.deepcopy(fixture.physics[1]);state['physics_step']=len(physics)
                physics.append(state);scorer.observe_physics(state)
            row=copy.deepcopy(fixture.rows[0]);row.update(step=i,time_s=(i+1)/20,physics=state)
            rows.append(row)
            online=scorer.observe_control(row)
            offline=task_score(rows,physics,cfg,fixture.meta,'running')
            self.assertEqual(online['task_score'],offline['task_score'])
            self.assertEqual(online['retained_pairs'],offline['retained_pairs'])
        self.assertEqual(scorer.result()['task_score'],1.)

    def test_incomplete_native_period_cannot_keep_a_previous_positive_score(self):
        fixture=test_tabletop.TabletopScoreChecks();fixture.setUp()
        scorer=StreamingScore(fixture.cfg,fixture.meta)
        scorer.observe_physics(fixture.physics[0])
        for i,row in enumerate(fixture.rows):
            scorer.observe_physics(fixture.physics[i+1]);scorer.observe_control(row)
        self.assertEqual(scorer.result()['task_score'],1.)
        state=copy.deepcopy(fixture.physics[-1]);state['physics_step']+=1
        scorer.observe_physics(state)
        self.assertEqual(scorer.finalize()['task_score'],0.)
        self.assertIn('incomplete_episode',scorer.result()['invalid_reasons'])

    def test_native_worker_failure_is_explicitly_zero_in_both_scorers(self):
        fixture=test_tabletop.TabletopScoreChecks();fixture.setUp()
        scorer=StreamingScore(fixture.cfg,fixture.meta)
        scorer.observe_physics(fixture.physics[0])
        for i,row in enumerate(fixture.rows):
            scorer.observe_physics(fixture.physics[i+1]);scorer.observe_control(row)
        fixture.meta['native_execution_error']=True
        self.assertEqual(scorer.finalize()['task_score'],0.)
        offline=task_score(fixture.rows,fixture.physics,fixture.cfg,fixture.meta,'worker_failure')
        self.assertEqual(offline['task_score'],0.)
        self.assertIn('native_execution_error',offline['invalid_reasons'])
