"""Reject release postures that previously tipped or snagged the assembled mop."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from robot_reopen import handle_pose
from supported_release import (handle_angles, handle_clear_of_left_exit, level_bridge_target,
                               open_hand_stationary, ready_to_release)


class SupportedReleaseTests(unittest.TestCase):
    def test_free_handle_must_point_away_from_the_left_exit(self):
        bridge = np.eye(4)
        bridge[:3, :3] = Rotation.from_euler('z', 19., degrees=True).as_matrix()
        for x, y, expected in [(32.3, 85., True), (-32.3, 85., False),
                                (0., 82., True), (32.3, 70., False),
                                (32.3, -82., False)]:
            with self.subTest(x=x, y=y):
                handle = handle_pose(bridge, np.deg2rad(x), np.deg2rad(y))
                self.assertEqual(handle_clear_of_left_exit(bridge, handle), expected)

    def test_opening_never_changes_an_arm_or_the_other_gripper_target(self):
        class Session:
            def __init__(self):
                self.action = np.linspace(-.3, .4, 14)
                self.action[6] = .08
                self.commands = []

            def step(self, command, phase):
                self.commands.append(command.copy())
                self.action = command.copy()

        session = Session()
        initial = session.action.copy()
        open_hand_stationary(session, 'left', .25, .4, 'opening')
        self.assertEqual(len(session.commands), 8)
        unchanged = np.arange(14) != 6
        for command in session.commands:
            np.testing.assert_array_equal(command[unchanged], initial[unchanged])
        apertures = np.array([initial[6], *[command[6] for command in session.commands]])
        self.assertTrue(np.all(np.diff(apertures) >= 0.))
        self.assertAlmostEqual(apertures[-1], .25)

    def test_post_release_leveling_preserves_contact_and_loaded_command_bias(self):
        bridge = np.eye(4)
        bridge[:3, :3] = Rotation.from_euler('xyz', [-7.2, .02, -5.7], degrees=True).as_matrix()
        bridge[:3, 3] = [-.08, -.03, .0102]
        grasp = np.eye(4)
        grasp[:3, :3] = np.diag([-1., 1., -1.])
        grasp[:3, 3] = [.030, .037, .033]
        measured = bridge @ grasp
        command = measured.copy()
        command[:3, 3] += [.002, -.001, .003]
        command[:3, :3] = Rotation.from_euler('x', 1., degrees=True).as_matrix() @ command[:3, :3]
        target = level_bridge_target(bridge, measured, command, 0.)
        moved_wrist = measured.copy()
        moved_wrist[:3, 3] += target[:3, 3] - command[:3, 3]
        moved_wrist[:3, :3] = target[:3, :3] @ command[:3, :3].T @ measured[:3, :3]
        moved_head = moved_wrist @ np.linalg.inv(grasp)
        np.testing.assert_allclose(moved_head[:2, 3], bridge[:2, 3], atol=1e-12)
        self.assertAlmostEqual(moved_head[2, 3], .005)
        np.testing.assert_allclose(moved_head[:3, 2], [0., 0., 1.], atol=1e-12)
        np.testing.assert_allclose(target[:3, 3] - moved_wrist[:3, 3],
                                   command[:3, 3] - measured[:3, 3], atol=1e-12)

    def test_post_release_leveling_rejects_large_motion(self):
        for height, tilt in ((.05, 0.), (.005, 25.)):
            bridge = np.eye(4)
            bridge[2, 3] = height
            bridge[:3, :3] = Rotation.from_euler('x', tilt, degrees=True).as_matrix()
            with self.subTest(height=height, tilt=tilt), self.assertRaises(AssertionError):
                level_bridge_target(bridge, np.eye(4), np.eye(4), 0.)

    def test_gimbal_measurement_uses_the_head_frame(self):
        bridge = np.eye(4)
        bridge[:3, :3] = Rotation.from_euler('xyz', [4., -3., 19.], degrees=True).as_matrix()
        handle = handle_pose(bridge, np.deg2rad(7.), np.deg2rad(82.))
        np.testing.assert_allclose(np.rad2deg(handle_angles(bridge, handle)), [7., 82.], atol=1e-10)

    def test_only_supported_lengthwise_release_is_allowed(self):
        for roll, height, x, y, expected in [
            (0., .005, 0., 82., True),
            (40., .035, 0., 82., False),
            (0., .005, 69., -10., False),
            (0., .005, 0., 0., False),
            (0., .020, 0., 82., False),
            (0., -.005, 0., 82., False),
        ]:
            with self.subTest(roll=roll, height=height, x=x, y=y):
                bridge = np.eye(4)
                bridge[:3, :3] = Rotation.from_euler('x', roll, degrees=True).as_matrix()
                bridge[2, 3] = height
                handle = handle_pose(bridge, np.deg2rad(x), np.deg2rad(y))
                self.assertEqual(ready_to_release(bridge, handle, 0.), expected)
