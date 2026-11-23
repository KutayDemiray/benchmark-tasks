"""Check vertical grasp geometry and rejection of an unstable release posture."""

from task_paths import OUTPUT_ROOT

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from robot_reopen import handle_pose
from upright_release import place_upright, upright_target


class UprightReleaseTests(unittest.TestCase):
    def test_vertical_target_preserves_contact_grasp_for_a_tilted_head(self):
        bridge = np.eye(4)
        bridge[:3, :3] = Rotation.from_euler('xyz', [3., -5., 17.], degrees=True).as_matrix()
        bridge[:3, 3] = [-.06, -.025, .34]
        handle = handle_pose(bridge, .2, 1.1)
        grasp = np.eye(4)
        grasp[:3, :3] = Rotation.from_euler('xyz', [30., 50., 20.], degrees=True).as_matrix()
        grasp[:3, 3] = [.003, -.004, .065]
        target = upright_target(bridge, handle, handle @ grasp)
        desired_handle = target @ np.linalg.inv(grasp)
        np.testing.assert_allclose(desired_handle[:3, 2], [0., 0., 1.], atol=1e-12)
        np.testing.assert_allclose(np.linalg.inv(desired_handle) @ target, grasp, atol=1e-12)

    def test_nonconverging_posture_stops_before_handle_release(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as directory:
            class Session:
                out = Path(directory)

                def __init__(self):
                    self.phases = []

                def move(self, *args):
                    self.phases.append(args[-1])

                def hold(self, seconds, phase):
                    self.phases.append(phase)

            session = Session()
            bridge = np.eye(4)
            bridge[2, 3] = .34
            handle = handle_pose(bridge, 0., np.deg2rad(8.))
            wrist = handle.copy()
            wrist[:3, 3] += handle[:3, 2] * .065
            with self.assertRaisesRegex(AssertionError, 'too tilted'):
                place_upright(session, bridge, handle, wrist, 0.,
                    lambda: {'Bridge': bridge, 'Handle': handle}, lambda side: wrist)
            self.assertEqual(session.phases.count('level_handle_before_release'), 6)
            self.assertNotIn('release_handle', session.phases)

    def test_vertical_handle_does_not_allow_a_tipped_head_to_be_released(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as directory:
            class Session:
                out = Path(directory)

                def __init__(self):
                    self.phases = []

                def move(self, *args):
                    self.phases.append(args[-1])

                def hold(self, seconds, phase):
                    pass

            session = Session()
            bridge = np.eye(4)
            bridge[:3, :3] = Rotation.from_euler('x', -40., degrees=True).as_matrix()
            bridge[2, 3] = .035
            handle = np.eye(4)
            handle[2, 3] = .085
            wrist = handle.copy()
            wrist[2, 3] += .065
            with self.assertRaisesRegex(AssertionError, 'too tilted or elevated'):
                place_upright(session, bridge, handle, wrist, 0.,
                    lambda: {'Bridge': bridge, 'Handle': handle}, lambda side: wrist)
            self.assertEqual(session.phases.count('level_handle_before_release'), 6)
            self.assertNotIn('release_handle', session.phases)


if __name__ == '__main__':
    unittest.main()
