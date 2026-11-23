"""Keep diagnostic initialization accurate and separate from benchmark scoring."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from release_probe import check_configuration, finger_joints_from_links


class ReleaseProbeTests(unittest.TestCase):
    def test_refinement_cannot_change_material_geometry_or_reduce_resolution(self):
        original = dict(physics_hz=480, iterations=64, contact_updates=8, contact_subiterations=2,
                        spacing=.0025, ccd=True, product_profile='base_micro', mouth_height=.016)
        for key, value in [('physics_hz', 240), ('iterations', 32), ('contact_updates', 4),
                           ('contact_subiterations', 1), ('spacing', .005), ('ccd', False),
                           ('product_profile', 'different'), ('mouth_height', .02)]:
            changed = dict(original, release_probe_refine_contact=True)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                check_configuration(original, changed)

    def test_numerical_refinement_is_explicit_and_records_every_difference(self):
        original = dict(physics_hz=480, iterations=64, contact_updates=8, contact_subiterations=2)
        changed = dict(original, physics_hz=960, contact_updates=16)
        with self.assertRaises(AssertionError):
            check_configuration(original, changed)
        changed['release_probe_refine_contact'] = True
        self.assertEqual(check_configuration(original, changed), {
            'physics_hz': {'original': 480, 'diagnostic': 960},
            'contact_updates': {'original': 8, 'diagnostic': 16}})
        with self.assertRaisesRegex(AssertionError, 'increased numerical setting'):
            check_configuration(original, dict(original, release_probe_refine_contact=True))

    def native_poses(self):
        from yam_arena import YAM_URDF
        from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
        from yam_arena.urdf import forward_kinematics

        joints = dict(zip(ARM_JOINTS, [.1, .8, 1.1, -.4, .15, -.1], strict=True))
        joints.update(zip(FINGER_JOINTS, [-.012, -.019], strict=True))
        links = forward_kinematics(YAM_URDF, joints)
        global_pose = np.eye(4)
        global_pose[:3, :3] = Rotation.from_euler('xyz', [13., -21., 41.], degrees=True).as_matrix()
        global_pose[:3, 3] = [.7, -.2, 1.1]
        names, poses = [], []
        for name, local in links.items():
            pose = global_pose @ local
            names.append(name)
            poses.append(np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()])
        return np.asarray(poses), names, {name: joints[name] for name in FINGER_JOINTS}

    def test_individual_fingers_survive_a_global_frame_change(self):
        poses, names, expected = self.native_poses()
        recovered = finger_joints_from_links(poses, names)
        for name, value in expected.items():
            self.assertAlmostEqual(recovered[name], value, places=12)

    def test_inconsistent_finger_geometry_is_rejected(self):
        poses, names, _ = self.native_poses()
        parent = Rotation.from_quat(poses[names.index('gripper'), 3:]).as_matrix()
        poses[names.index('tip_left'), :3] += parent[:, 1] * .01
        with self.assertRaisesRegex(AssertionError, 'prismatic geometry'):
            finger_joints_from_links(poses, names)

    def test_probe_cannot_enter_benchmark_scoring(self):
        from score_full_run import score

        request = '{"robot_cover": true, "release_probe_run": "recorded-source"}'
        with patch.object(Path, 'read_text', return_value=request):
            with self.assertRaisesRegex(AssertionError, 'Release probes cannot score'):
                score(Path('/unused-diagnostic-path'))
