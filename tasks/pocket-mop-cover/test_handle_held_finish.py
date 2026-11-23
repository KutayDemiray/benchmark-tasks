"""Versioned finish checks: exclusive handle support, retained lift and invalidity."""

from dataclasses import replace
import unittest

from handle_held_finish import HandleHeldScorer, HandleSupport
from score_logic import Observation


class HandleHeldTests(unittest.TestCase):
    def setUp(self):
        self.scorer = HandleHeldScorer()
        self.observation = Observation(0, .05, (.05, .05), (.9, .9), True, 0.,
            False, False, .12, (.3, .1), .13, 0., handle_grasped=True)
        self.support = HandleSupport('right', .025, .04, (0., 0.))

    def hold(self, count=40, support=None, **kwargs):
        for _ in range(count):
            observation = replace(self.observation, step=self.scorer.base.last_step+1, **kwargs)
            result = self.scorer.update(observation, self.support if support is None else support)
        return result

    def sequence(self, support=None, **kwargs):
        self.hold(40, support, **kwargs)
        self.hold(1, support, head_xy_m=(.19, .1), **kwargs)
        return self.hold(40, support, **kwargs)

    def test_full_reward_with_handle_held_without_setdown(self):
        result = self.sequence()
        self.assertEqual(result['score'], 1.)
        self.assertTrue(result['success'])
        self.scorer.reset()
        self.assertEqual(self.hold(1)['score'], 0.)

    def test_handle_proximity_alone_cannot_score_full_reward(self):
        self.assertEqual(self.hold(100)['score'], .5)
        self.assertFalse(self.scorer.base.retention_verified)

    def test_bridge_latch_or_other_hand_support_resets_functional_proof(self):
        for bad in (replace(self.support, non_handle_robot_clearance_m=.0019),
                    replace(self.support, other_robot_clearance_m=.0199),
                    replace(self.support, finger_handle_clearance_m=(0., .003)), HandleSupport()):
            self.scorer.reset()
            self.assertEqual(self.sequence(support=bad)['score'], .5)
            self.assertFalse(self.scorer.base.retention_verified)

    def test_wrong_pocket_and_unlocked_wings_cannot_receive_full_reward(self):
        for change in ({'pocket_depth_m': (.05, 0.)}, {'orientation_correct': False},
                       {'wing_angle_deg': 4.}, {'textile_grasped': True},
                       {'wings_externally_supported': True}, {'all_cloth_clearance_m': .099}):
            self.scorer.reset()
            self.assertLess(self.sequence(**change)['score'], 1.)

    def test_brief_support_or_touchdown_requires_new_motion_check(self):
        self.assertTrue(self.sequence()['success'])
        self.hold(1, support=replace(self.support, other_robot_clearance_m=0.))
        self.assertEqual(self.hold()['score'], .5)
        self.assertTrue(self.sequence()['success'])
        self.hold(1, all_cloth_clearance_m=.05)
        self.assertEqual(self.hold()['score'], .5)

    def test_changing_hands_requires_new_motion_check(self):
        self.assertTrue(self.sequence()['success'])
        self.hold(1, support=replace(self.support, side='left'))
        self.assertEqual(self.hold(support=replace(self.support, side='left'))['score'], .5)

    def test_numerical_failures_remain_disqualifying(self):
        for bad in ({'max_edge_strain': .201}, {'max_penetration_m': .0011},
                    {'unexpected_constraint': True}, {'diagnostic_fixture': True}):
            self.scorer.reset()
            self.hold(1, **bad)
            self.assertEqual(self.sequence()['score'], 0.)
        self.scorer.reset()
        self.hold(1, support=replace(self.support, other_robot_clearance_m=float('nan')))
        self.assertEqual(self.sequence()['score'], 0.)


class HandleSupportGeometryTests(unittest.TestCase):
    def fixture(self):
        import itertools
        import numpy as np
        from handle_held_finish import HandleSupportObserver
        corners = np.array(list(itertools.product((-.005, .005), repeat=3)))
        frame = dict(colliders=[dict(body=name, points=corners) for name in ('Handle', 'Bridge', 'Rocker')])
        names = dict(body_names=['Handle', 'Bridge', 'Rocker'])
        robots = {side: dict(body_names=['tip_left', 'tip_right', 'link1'],
            colliders=[dict(body=name, points=corners) for name in ('tip_left', 'tip_right', 'link1')])
            for side in ('left', 'right')}
        def poses(points):
            return np.repeat(np.array([[*point, 0., 0., 0., 1.] for point in points])[None], 3, axis=0)
        state = dict(frame_body_poses=poses([[0., 0., 0.], [0., 0., -.2], [0., .02, -.2]]),
                     left_body_poses=poses([[1., 0., 0.], [1., .02, 0.], [1., .2, 0.]]),
                     right_body_poses=poses([[.01, 0., 0.], [-.01, 0., 0.], [.1, 0., 0.]]),
                     positions=np.repeat(np.array([[[-.1, 0., -.2], [.1, 0., -.2], [0., .1, -.2]]]), 3, axis=0))
        return HandleSupportObserver(frame, names, robots, np.array([[0, 1, 2]])), state

    def test_contact_only_on_handle_and_entire_other_arm_clear(self):
        observer, state = self.fixture()
        support, _ = observer.measure(state)
        self.assertTrue(support.valid())
        self.assertEqual(support.side, 'right')

    def test_middle_native_state_can_interrupt_handle_grip(self):
        observer, state = self.fixture()
        state['right_body_poses'][1, 0, 0] += .04
        support, _ = observer.measure(state)
        self.assertFalse(support.valid())

    def test_middle_native_state_bridge_contact_invalidates_handle_only_support(self):
        observer, state = self.fixture()
        state['right_body_poses'][1, 2, :3] = [0., 0., -.2]
        support, _ = observer.measure(state)
        self.assertFalse(support.valid())

    def test_free_arm_link_contact_invalidates_otherwise_clear_grippers(self):
        observer, state = self.fixture()
        state['left_body_poses'][1, 2, :3] = [0., 0., -.2]
        support, _ = observer.measure(state)
        self.assertFalse(support.valid())


if __name__ == '__main__':
    unittest.main()
