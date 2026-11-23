"""CPU regressions for the observed wrist-limit failure; not contact validation."""
import unittest
import numpy as np
from yam_arena import ik
from motion import Arm, orientation
from controller import release, working_grasp
from mechanism import load_spec, transform, link_transforms, jaw_angle_for_gap


class MotionTests(unittest.TestCase):
    def right_at_observed_release(self):
        arm = Arm("right")
        arm.q = np.array([.94142014, 2.07301744, .53287272, 1.5419531, .95969982, 1.5622324])
        arm.pos, arm.rot = ik.tcp_pose(arm.q, arm.base)
        arm.grip = .10767
        return arm

    def test_clear_then_rise_resolves_observed_retreat_failure(self):
        old = self.right_at_observed_release()
        with self.assertRaises(RuntimeError):
            list(old.move(old.pos-old.rot[:, 2]*.07, old.rot, 1.5))
        arm = self.right_at_observed_release()
        start = arm.pos.copy()
        for _ in release(arm):
            self.assertTrue(np.all(arm.q >= ik.LOWER) and np.all(arm.q <= ik.UPPER))
        self.assertGreater(arm.pos[2]-start[2], .038)
        self.assertGreater(start[0]-arm.pos[0], .038)

    def test_free_width_approach_stays_inside_stock_limits(self):
        arm = self.right_at_observed_release()
        list(release(arm))
        target = np.array([.347, -.102, .89])
        rot = orientation([1, 0, 0], [0, .6, -.8])
        for _ in arm.move_free(target, rot, 3., grip=.35):
            self.assertTrue(np.all(arm.q >= ik.LOWER) and np.all(arm.q <= ik.UPPER))
        self.assertLess(np.linalg.norm(arm.pos-target), .001)
        self.assertLess(np.linalg.norm(ik._rot_error(rot, arm.rot)), np.deg2rad(1))

    def test_working_grasps_allow_tip_down_jaw_operation(self):
        # The previous sideways pinch could reach a closed tip-down pose, but
        # exceeded the left wrist limit as soon as the deployed jaw opened.
        spec = load_spec()
        root = transform([.24, 0., .812], np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]))
        for side, body in (("left", "handle_b"), ("right", "handle_a")):
            arm = Arm(side)
            for gap in np.linspace(.0007, .016, 25):
                q = dict(jaw=jaw_angle_for_gap(spec, gap), fold_a=0., fold_b=0.)
                tr = root@link_transforms(spec, q)[body]@working_grasp(body)
                arm.aim(tr[:3, 3], tr[:3, :3])
                self.assertTrue(np.all(arm.q >= ik.LOWER) and np.all(arm.q <= ik.UPPER))


if __name__ == "__main__":
    unittest.main()
