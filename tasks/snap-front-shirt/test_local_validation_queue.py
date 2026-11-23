"""Keep automatic continuation local, bounded and behind the physical gate."""

from pathlib import Path
import unittest
from unittest.mock import patch

from local_validation_queue import Queue,launch_command,retained_pilot,stable_gate


class QueueChecks(unittest.TestCase):
    def test_all_native_modes_use_exclusive_stable_id(self):
        for mode in ('hold','bimanual','ablation','scripted','action_replay'):
            command=launch_command(Path('/bundle'),Path('/out'),Path('/gate'),mode,1)
            self.assertEqual(command[command.index('--task')+1],'snap-front-shirt')
            self.assertEqual(command[command.index('--mode')+1],'exclusive')
            self.assertIn('evaluation',command)
        for forbidden in ('dream','model','fake','upload'):
            with self.assertRaises(AssertionError):launch_command(Path('/b'),Path('/o'),Path('/g'),forbidden,1)

    def test_failed_or_unaudited_gate_cannot_start_a_native_job(self):
        for result in ({},{'gate_passed':True},{'verification_pass':True,'gate_passed':False}):
            self.assertFalse(stable_gate(result))
            queue=Queue(Path('/not-written'),Path('/gate'))
            with patch('review_stability_round.review',return_value=result),patch('local_validation_queue.write'), \
                 patch.object(queue,'state'),patch.object(queue,'job') as job:
                queue.continue_after_gate();job.assert_not_called()

    def test_partial_pilot_requires_valid_released_pair_and_contact_guard(self):
        good=dict(retained_pairs=[0],invalid_reasons=[],pilot_rigid_contact_guard_passed=True)
        self.assertTrue(retained_pilot(good))
        for change in ({'retained_pairs':[]},{'invalid_reasons':['cloth_strain_limit']},
                       {'pilot_rigid_contact_guard_passed':False}):
            self.assertFalse(retained_pilot(dict(good,**change)))
