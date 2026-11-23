"""Check mass and inertia preservation in the diagnostic compound control."""

from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from passive_contact import check_request, compound_properties, measured_joint_angles, partition_readback, redundant_wing_limit_proof, native_physics_errors, rebase_body_origins


class PassiveContactTests(unittest.TestCase):
    def test_parallel_axis_theorem(self):
        poses = np.array([[-1., 0., 0., 0., 0., 0., 1.], [1., 0., 0., 0., 0., 0., 1.]])
        masses = np.array([2., 2.])
        coms = np.zeros((2, 7)); coms[:, 6] = 1
        inertias = np.array([np.eye(3), np.eye(3)])
        mass, center, inertia, _, _ = compound_properties(poses, masses, coms, inertias)
        self.assertEqual(mass, 4.)
        np.testing.assert_allclose(center, [1., 0., 0.])
        np.testing.assert_allclose(inertia, np.diag([2., 6., 6.]))

    def test_common_world_transform_does_not_change_body_properties(self):
        poses = np.array([[.1, .2, .3, 0., 0., 0., 1.], [.8, -.2, .5, .1, .2, .3, .9]])
        poses[:, 3:] = Rotation.from_quat(poses[:, 3:]).as_quat()
        masses = np.array([.3, .7])
        coms = np.array([[.01, .02, .03, 0, 0, 0, 1], [.03, .04, .05, 0, 0, 0, 1]])
        inertias = np.array([np.diag([.01, .02, .03]), np.diag([.02, .03, .04])])
        rotation = Rotation.from_euler('xyz', [23., -32., 68.], degrees=True)
        transformed = poses.copy()
        transformed[:, :3] = rotation.apply(poses[:, :3])+[4., -2., 7.]
        transformed[:, 3:] = (rotation*Rotation.from_quat(poses[:, 3:])).as_quat()
        first = compound_properties(poses, masses, coms, inertias)
        second = compound_properties(transformed, masses, coms, inertias)
        for before, after in zip(first, second, strict=True):
            np.testing.assert_allclose(before, after, atol=1e-13)

    def test_counterfactual_cannot_enter_an_ordinary_episode(self):
        for request in ({'experiment': 'frame_robot', 'passive_mode': 'compound'},
                        {'experiment': 'frame_robot', 'diagnostic_output_root': '/unused'},
                        {'experiment': 'frame_robot', 'passive_source': 'existing'},
                        {'experiment': 'frame_robot', 'passive_canonical_order': True},
                        {'experiment': 'frame_robot', 'passive_pvd_record': True},
                        {'experiment': 'frame_robot', 'passive_match_kit_defaults': True},
                        {'experiment': 'frame_robot', 'passive_stop_when_settled': True},
                        {'experiment': 'frame_robot', 'passive_cloth_correction': 'bounded'},
                        {'experiment': 'frame_robot', 'passive_cloth_correction': 'unbounded'},
                        {'experiment': 'frame_robot', 'passive_sdk_scene': 'existing'},
                        {'experiment': 'frame_robot', 'passive_host_readback': True},
                        {'experiment': 'frame_robot', 'passive_usd_readback': True},
                        {'experiment': 'frame_robot', 'passive_native_pipeline': True},
                        {'experiment': 'frame_robot', 'passive_handle_roll': 0.},
                        {'experiment': 'frame_robot', 'passive_solver': 'PGS'},
                        {'experiment': 'frame_robot', 'passive_partitions': 1},
                        {'experiment': 'frame_robot', 'passive_external_forces_once': True},
                        {'experiment': 'frame_robot', 'passive_no_self_contact': True},
                        {'experiment': 'frame_robot', 'passive_contact_subiterations': 1},
                        {'experiment': 'frame_robot', 'passive_no_joint_friction': True},
                        {'experiment': 'frame_robot', 'passive_deduplicate_wing_limits': True},
                        {'experiment': 'frame_robot', 'passive_mimic_frequency': 0.},
                        {'experiment': 'frame_robot', 'passive_mimic_frequency': 480.},
                        {'experiment': 'frame_robot', 'passive_free_space': True},
                        {'experiment': 'frame_robot', 'passive_zero_bending': True},
                        {'experiment': 'frame_robot', 'passive_center_body_frames': True},
                        {'experiment': 'frame_robot', 'passive_position_iterations': 4},
                        {'experiment': 'frame_robot', 'passive_head_correction': True},
                        {'experiment': 'frame_robot', 'passive_release_height': .01},
                        {'experiment': 'frame_robot', 'passive_no_contact_friction': True},
                        {'experiment': 'frame_robot', 'passive_freeze_joints': 'handle'},
                        {'experiment': 'frame_robot', 'passive_freeze_joints': 'head'},
                        {'experiment': 'frame_robot', 'passive_freeze_joints': 'all'},
                        {'experiment': 'frame_robot', 'passive_unit_scale': 10.},
                        {'experiment': 'frame_robot', 'passive_no_frame_self_contact': True}):
            with self.subTest(request=request), self.assertRaises(AssertionError):
                check_request(request, Path('/unused'))
        check_request({'experiment': 'frame_robot'}, Path('/unused'))

    def test_standalone_cannot_silently_change_source_gravity(self):
        import json
        from tempfile import TemporaryDirectory
        from standalone_contact import validate_source
        request = dict(experiment='passive_contact', passive_sdk_scene='source',
            passive_source='held', passive_mode='articulated', passive_free_space=True,
            physics_hz=480, iterations=64, spacing=.0025, contact_updates=8,
            contact_subiterations=2, ccd=True, product_profile='base_micro')
        original = dict(request, passive_sdk_scene=None)
        with TemporaryDirectory() as folder:
            output = Path(folder)
            source = output/'source'
            source.mkdir()
            (source/'result.json').write_text(json.dumps(dict(status='completed')))
            (source/'native-initialization-check.json').write_text(json.dumps(dict(passed=True)))
            (source/'kit.log').write_text('')
            (source/'request.json').write_text(json.dumps(original))
            self.assertEqual(validate_source(request, output), source)
            request['passive_free_space'] = False
            with self.assertRaisesRegex(AssertionError, 'gravity control'):
                validate_source(request, output)

    def test_joint_angles_are_relative_to_moving_parent(self):
        parent = Rotation.from_euler('xyz', [42., -13., 73.], degrees=True)
        child = parent*Rotation.from_euler('y', -27., degrees=True)
        poses = np.zeros((2, 7)); poses[:, 3:] = [parent.as_quat(), child.as_quat()]
        angle = measured_joint_angles(poses, ['parent', 'child'], [dict(parent='parent', child='child', axis='Y')])
        np.testing.assert_allclose(angle, [np.deg2rad(-27.)], atol=1e-14)

    def test_partitioned_readback_rejects_duplicate_or_missing_bodies(self):
        from types import SimpleNamespace
        def part(names):
            return SimpleNamespace(body_names=names, root_view=SimpleNamespace(get_masses=lambda: np.array([[1.]*len(names)])))
        np.testing.assert_allclose(partition_readback([part(['a']), part(['b'])], ['b', 'a'], 'get_masses'), [1., 1.])
        for parts in ([part(['a']), part(['a'])], [part(['a'])]):
            with self.assertRaises(AssertionError):
                partition_readback(parts, ['a', 'b'], 'get_masses')

    def test_redundant_wing_limits_preserve_the_entire_feasible_interval(self):
        proof = redundant_wing_limit_proof([-90., 0.], [0., 90.], -1., 0.)
        self.assertTrue(proof['physical_feasible_angles_unchanged'])
        for coefficient, offset in [(-2., 0.), (-1., .1), (1., 0.)]:
            with self.assertRaisesRegex(AssertionError, 'feasible mechanism'):
                redundant_wing_limit_proof([-90., 0.], [0., 90.], coefficient, offset)

    def test_parser_rejection_is_not_mistaken_for_accepted_native_physics(self):
        errors = [
            '[Error] [omni.physx.plugin] Usd Physics: NewtonMimicAPI follower requires finite limits.',
            'PhysX error: gear constraint unsupported',
            '[ERROR] [omni.physx.sdk] Native solver rejected the configuration',
            '[Error] [omni.physics.tensors.plugin] Could not read native state',
        ]
        warnings = ['[Warning] [omni.physx.plugin] Diagnostic notice', '[Error] [omni.renderer.plugin] Unrelated renderer error']
        self.assertEqual(native_physics_errors('\n'.join(errors+warnings)), errors)

    def test_body_recentering_preserves_world_geometry_and_round_trip(self):
        pose = np.array([[.2, -.4, .6, *Rotation.from_euler('xyz', [20., -73., 6.], degrees=True).as_quat()]])
        offset = np.array([[.11, -.03, .07]])
        point = np.array([[.3, .2, -.1]])
        rebased = rebase_body_origins(pose, offset)
        original_world = pose[:, :3]+Rotation.from_quat(pose[:, 3:]).apply(point)
        rebased_world = rebased[:, :3]+Rotation.from_quat(rebased[:, 3:]).apply(point-offset)
        np.testing.assert_allclose(rebased_world, original_world, atol=1e-14)
        np.testing.assert_allclose(rebase_body_origins(rebased, offset, inverse=True), pose, atol=1e-14)

    def test_unit_conversion_preserves_rotations_and_angular_velocity(self):
        from passive_units import pose_to_metres, velocity_to_metres
        np.testing.assert_allclose(pose_to_metres([[10., 20., 30., 0., 0., 0., 1.]], 100.),
            [[.1, .2, .3, 0., 0., 0., 1.]])
        np.testing.assert_allclose(velocity_to_metres([[10., 20., 30., 1., 2., 3.]], 100.),
            [[.1, .2, .3, 1., 2., 3.]])


    def test_centered_release_keeps_head_and_hinge_anchors_unchanged(self):
        from passive_contact import centered_handle_state
        names = ['Bridge', 'Connector', 'Handle', 'Left']
        pose = np.zeros((4, 7)); pose[:, 6] = 1.
        pose[0] = [.2, .4, .005, *Rotation.from_euler('z', 27., degrees=True).as_quat()]
        before = pose.copy()
        q = np.deg2rad([33., 85., -1.])
        after, angles = centered_handle_state(pose, q, names, ['HandleX', 'HandleY', 'HingeLeft'], 0.)
        np.testing.assert_array_equal(after[[0, 3]], before[[0, 3]])
        np.testing.assert_array_equal(pose, before)
        np.testing.assert_array_equal(angles[1:], q[1:])
        bridge_rotation = Rotation.from_quat(before[0, 3:])
        np.testing.assert_allclose(after[1, :3], before[0, :3]+bridge_rotation.apply([.030, 0., .052]))
        np.testing.assert_allclose(after[2, :3], after[1, :3]+bridge_rotation.apply([0., 0., .010]))
        relative = bridge_rotation.inv()*Rotation.from_quat(after[2, 3:])
        np.testing.assert_allclose(relative.as_rotvec(), [0., q[1], 0.], atol=1e-14)
