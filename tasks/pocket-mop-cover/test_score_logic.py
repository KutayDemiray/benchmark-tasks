"""Physical-sequence positive and shortcut-negative rubric tests."""
from dataclasses import replace
import unittest
from score_logic import Observation, OutcomeScorer


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.scorer = OutcomeScorer()
        self.nominal = Observation(0, .05, (.05, .05), (.9, .9), True, 0.,
                                   False, False, 0., (.3, .1), .006, .03)

    def hold(self, count=40, **kwargs):
        for _ in range(count):
            obs = replace(self.nominal, step=self.scorer.last_step + 1, **kwargs)
            result = self.scorer.update(obs)
        return result

    def retention(self):
        self.hold()
        self.hold(all_cloth_clearance_m=.12, head_height_over_support_m=.13)
        self.hold(1, all_cloth_clearance_m=.12, head_xy_m=(.3, -.01),
                  head_height_over_support_m=.13)
        return self.hold(all_cloth_clearance_m=.12, head_height_over_support_m=.13)

    def test_full_sequence_and_reset(self):
        self.retention()
        result = self.hold()
        self.assertEqual(result["score"], 1.)
        self.scorer.reset()
        self.assertEqual(self.hold(1)["score"], 0.)
        self.assertFalse(self.scorer.retention_verified)

    def test_one_pocket_only(self):
        result = self.hold(pocket_depth_m=(.05, 0.))
        self.assertEqual(result["score"], .25)

    def test_uninstalled_pocket_orientation_does_not_cancel_valid_other_pocket(self):
        result=self.hold(orientation_correct=False,pocket_orientation_correct=(True,False))
        self.assertEqual(result['score'],.25)
        self.assertFalse(result['retention_verified'])
        self.scorer.reset()
        result=self.hold(orientation_correct=False,pocket_orientation_correct=(False,True))
        self.assertEqual(result['score'],.25)

    def test_on_top_or_wrong_orientation(self):
        self.assertEqual(self.hold(pocket_width_fraction=(0., 0.))["score"], 0.)
        self.assertEqual(self.hold(orientation_correct=False)["score"], 0.)

    def test_gripper_supported_installation(self):
        self.assertEqual(self.hold(textile_grasped=True)["score"], 0.)

    def test_flat_is_not_a_lock_demonstration(self):
        self.assertEqual(self.hold(100)["score"], .5)
        self.assertFalse(self.scorer.retention_verified)

    def test_wing_support_prevents_retention(self):
        self.retention()
        result = self.hold(wings_externally_supported=True)
        self.assertEqual(result["score"], .5)
        self.assertFalse(result["retention_verified"])

    def test_setdown_and_clearance_required(self):
        self.retention()
        self.assertEqual(self.hold(all_grippers_clearance_m=.005)["score"], .5)
        self.assertEqual(self.hold(head_height_over_support_m=.1)["score"], .5)

    def test_handle_grasp_is_allowed_for_retention_but_not_final_release(self):
        self.hold(handle_grasped=True)
        self.hold(all_cloth_clearance_m=.12,head_height_over_support_m=.13,handle_grasped=True)
        self.hold(1,all_cloth_clearance_m=.12,head_height_over_support_m=.13,
                  head_xy_m=(.3,-.01),handle_grasped=True)
        result=self.hold(all_cloth_clearance_m=.12,head_height_over_support_m=.13,handle_grasped=True)
        self.assertTrue(result['retention_verified'])
        self.assertEqual(self.hold(handle_grasped=True)['score'],.5)
        self.assertEqual(self.hold()['score'],1.)

    def test_placing_on_floor_or_outside_table_cannot_complete(self):
        self.retention()
        self.assertEqual(self.hold(head_height_over_support_m=-.74)['score'],.5)
        self.assertEqual(self.hold(over_support_surface=False)['score'],.5)
        self.assertEqual(self.hold()['score'],1.)

    def test_release_requires_both_native_and_difference_settling(self):
        self.retention()
        self.assertEqual(self.hold(max_cloth_speed_m_s=.031)["score"], .5)
        self.assertEqual(self.hold(max_cloth_difference_speed_m_s=.031)["score"], .5)
        self.assertEqual(self.hold()["score"], 1.)

    def test_invalidity_is_latched_until_reset(self):
        for violation in ({"max_penetration_m": .0011}, {"max_seam_gap_m": .0011},
                          {"max_edge_strain": .201}, {"finite": False},
                          {"unexpected_constraint": True}, {"diagnostic_fixture": True}):
            self.scorer.reset()
            self.hold(1, **violation)
            self.retention()
            self.assertEqual(self.hold()["score"], 0.)
        self.scorer.reset()
        self.retention()
        self.assertEqual(self.hold()["score"], 1.)

    def test_loss_of_pocket_clears_completed_retention(self):
        self.retention()
        self.hold(1, pocket_depth_m=(0., .05))
        self.assertEqual(self.hold()["score"], .5)

    def test_separate_short_lifts_cannot_accumulate(self):
        self.hold()
        for _ in range(3):
            self.hold(10, all_cloth_clearance_m=.12)
            self.hold(10)
        self.assertEqual(self.hold()["score"], .5)

    def test_observation_order_is_enforced(self):
        self.hold(1)
        with self.assertRaises(AssertionError):
            self.scorer.update(self.nominal)


if __name__ == "__main__":
    unittest.main()
