"""Independent load-error cases for ordinary wrist command correction."""

from task_paths import OUTPUT_ROOT

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from wrist_tracking import MeasuredWristGuide


class FakeSession:
    def __init__(self, out, offset, rotation):
        self.out = out
        self.offset = offset
        self.rotation = rotation

    def move_both(self, targets, seconds, phase):
        self.targets = targets

    def measure(self, side):
        position, rotation, _ = self.targets[side]
        pose = np.eye(4)
        pose[:3, 3] = position + self.offset
        pose[:3, :3] = self.rotation @ rotation
        return pose


class WristTrackingTests(unittest.TestCase):
    def test_body_marker_feedback_accounts_for_grasp_rotation(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as tmp:
            session=FakeSession(Path(tmp),np.array([0.,0.,-.02]),np.eye(3))
            slip=np.eye(4)
            slip[:3,:3]=Rotation.from_euler('y',3.,degrees=True).as_matrix()
            slip[:3,3]=[.002,0.,0.]
            def marker(side):
                return session.measure(side)@slip
            guide=MeasuredWristGuide(session,marker,session.measure,'native_held_body_markers')
            targets={side:(np.array([.1,.2,.3]),np.eye(3),.08) for side in ('left','right')}
            guide.settle(targets,'align_body_markers')
            for side in targets:
                self.assertLess(np.linalg.norm(marker(side)[:3,3]-targets[side][0]),.001)
                self.assertLess(Rotation.from_matrix(marker(side)[:3,:3]).magnitude(),.01)
                self.assertGreater(Rotation.from_matrix(session.measure(side)[:3,:3]).magnitude(),np.deg2rad(2.))
                self.assertGreater(np.linalg.norm(guide.records[-1]['hands'][side]['robot_rotation_error_rad']),np.deg2rad(2.))

    def test_alignment_waits_for_measured_convergence(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as tmp:
            session=FakeSession(Path(tmp),np.array([0.,0.,-.026]),
                Rotation.from_euler('y',7.,degrees=True).as_matrix())
            guide=MeasuredWristGuide(session,session.measure)
            targets={side:(np.array([.1,.2,.3]),np.eye(3),.08) for side in ('left','right')}
            guide.settle(targets,'align')
            self.assertGreater(len(guide.records),3)
            self.assertLessEqual(len(guide.records),20)
            for side in targets:
                self.assertLess(np.linalg.norm(session.measure(side)[:3,3]-targets[side][0]),.001)

    def test_constant_loaded_pose_error_converges(self):
        # A measurement bias stands in for physical compliance; the controller
        # has no access to this bias or to an object-state mutation interface.
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as tmp:
            session = FakeSession(Path(tmp), np.array([.003, -.002, -.012]),
                                  Rotation.from_euler('x', 5., degrees=True).as_matrix())
            guide = MeasuredWristGuide(session, session.measure)
            targets = {side: (np.array([.1, .2, .3]), np.eye(3), .08) for side in ('left', 'right')}
            for _ in range(7):
                guide.move(targets, .75, 'calibrate')
            for side in targets:
                measured = session.measure(side)
                self.assertLess(np.linalg.norm(measured[:3, 3] - targets[side][0]), .0001)
                self.assertLess(Rotation.from_matrix(measured[:3, :3]).magnitude(), .001)
                self.assertEqual(session.targets[side][2], .08)

    def test_unreachable_error_cannot_wind_up_without_bound(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as tmp:
            session = FakeSession(Path(tmp), np.ones(3), np.eye(3))
            def blocked_measure(side):
                pose = np.eye(4)
                pose[:3, 3] = [1., 1., 1.]
                pose[:3, :3] = Rotation.from_euler('x', 90., degrees=True).as_matrix()
                return pose
            guide = MeasuredWristGuide(session, blocked_measure)
            targets = {side: (np.zeros(3), np.eye(3), .25) for side in ('left', 'right')}
            for _ in range(30):
                guide.move(targets, .75, 'blocked')
            for side in targets:
                self.assertLessEqual(np.linalg.norm(guide.offset[side]), .045000001)
                self.assertLessEqual(Rotation.from_matrix(guide.rotation[side]).magnitude(), np.deg2rad(15.) + 1e-9)
            with self.assertRaises(AssertionError):
                guide.settle(targets,'blocked_alignment')


if __name__ == '__main__':
    unittest.main()
