"""Sequence-level score counterexamples and actual-size access geometry."""
import unittest
import numpy as np
from scipy.spatial import ConvexHull
from yam_arena import ik
from mechanism import load_spec, link_transforms, jaw_angle_for_gap, transform, rz
from load_controller import jaw_mid_local, TIP_DOWN
from retrieval_scene import (nut_spec, fixtures, POCKET_XY, POCKET_FLOOR,
                             POCKET_RIM, TRAY_XY, TRAY_FLOOR, NUT_HEIGHT, NUT_MASS)
from retrieval_score import RetrievalGate


def measurement(step, xy, bottom, pinch=True):
    spec = load_spec()
    angle = jaw_angle_for_gap(spec, .010)
    rot = TIP_DOWN@rz(-angle/2)
    center = np.array([*xy, bottom+NUT_HEIGHT/2])
    root = transform(center-rot@jaw_mid_local(spec, angle), rot)
    poses, forces, points = {}, [], []
    links = link_transforms(spec, dict(jaw=angle, fold_a=0., fold_b=0.))
    for body, sign in (("jaw_a", 1), ("jaw_b", -1)):
        tr = root@links[body]
        poses[body] = np.r_[tr[:3, 3], ik.matrix_to_quat_xyzw(tr[:3, :3])].tolist()
        forces.append((-sign*2*tr[:3, 1] if pinch else np.zeros(3)).tolist())
        points.append((tr@[.04775, sign*.00036, 0., 1.])[:3].tolist())
    half = np.array([.005774, .005, NUT_HEIGHT/2])
    low, high = center-half, center+half
    return dict(step=step, link_poses=poses, payload=dict(name="nut", pose=[*xy, bottom, 0., 0., 0., 1.],
        low=low.tolist(), high=high.tolist(), velocity=[0.]*6,
        contact_forces_n=forces+[[0.]*3 for _ in range(4)],
        contact_points_w=points+[[None]*3 for _ in range(4)],
        unattributed_normal_n=[0.]*3, tool_robot_clearance_m=.03 if not pinch else 0.))


def extracted(gate, variant=None):
    gate.update(measurement(0, POCKET_XY, POCKET_FLOOR, False), False)
    gate.update(measurement(1, POCKET_XY, POCKET_FLOOR), True)
    for i in range(20):
        m = measurement(i+2, POCKET_XY, POCKET_RIM+.011)
        if variant == "single_jaw":
            m["payload"]["contact_forces_n"][1] = [0.]*3
        elif variant == "robot_support":
            m["payload"]["contact_forces_n"][4] = [0., 0., .2]
        elif variant == "light_robot_support":
            # Even the taller nut weighs less than the old 0.1 N exclusion
            # threshold; a finger could otherwise carry its entire weight.
            m["payload"]["contact_forces_n"][4] = [0., 0., NUT_MASS*9.81]
        elif variant == "distributed_support":
            m["payload"]["contact_forces_n"][4] = [0., 0., .0025]
            m["payload"]["contact_forces_n"][5] = [0., 0., .0025]
        elif variant == "resting_on_support":
            m["payload"]["unattributed_normal_n"] = [0., 0., NUT_MASS*9.81]
        gate.update(m, True)
    gate.update(measurement(22, POCKET_XY+[.04, 0.], POCKET_RIM+.011), True)


class RetrievalTests(unittest.TestCase):
    def test_complete_chain_and_settle_duration(self):
        gate = RetrievalGate()
        extracted(gate)
        self.assertTrue(gate.carried)
        gate.update(measurement(23, TRAY_XY, TRAY_FLOOR+.01), True)
        for i in range(39):
            gate.update(measurement(24+i, TRAY_XY, TRAY_FLOOR, False), True)
        self.assertFalse(gate.passed)
        gate.update(measurement(63, TRAY_XY, TRAY_FLOOR, False), True)
        self.assertTrue(gate.passed)
        gate.update(measurement(64, POCKET_XY, POCKET_FLOOR, False), True)
        self.assertFalse(gate.passed, "End scoring must revoke a placement that no longer holds")

    def test_movement_without_qualified_extraction_is_insufficient(self):
        for variant in ("single_jaw", "robot_support", "light_robot_support", "distributed_support", "resting_on_support", "no_chain"):
            gate = RetrievalGate()
            if variant != "no_chain":
                extracted(gate, variant)
            else:
                gate.update(measurement(0, POCKET_XY, POCKET_FLOOR, False), False)
            gate.update(measurement(23, TRAY_XY, TRAY_FLOOR+.01), True)
            for i in range(45):
                gate.update(measurement(24+i, TRAY_XY, TRAY_FLOOR, False), True)
            self.assertTrue(gate.in_tray)
            self.assertFalse(gate.passed, variant)

    def test_lost_carry_cannot_be_repaired_by_direct_placement(self):
        gate = RetrievalGate()
        extracted(gate)
        gate.update(measurement(23, POCKET_XY+[.06, 0.], .7475, False), True)
        for i in range(45):
            gate.update(measurement(24+i, TRAY_XY, TRAY_FLOOR, False), True)
        self.assertTrue(gate.carried)
        self.assertFalse(gate.passed)

    def test_pinch_may_start_inside_recess_but_not_above_rim(self):
        for start_z, expected in ((POCKET_FLOOR+.003, True), (POCKET_RIM+.011, False)):
            gate = RetrievalGate()
            gate.update(measurement(0, POCKET_XY, POCKET_FLOOR, False), False)
            gate.update(measurement(1, POCKET_XY, start_z), True)
            for i in range(20):
                gate.update(measurement(i+2, POCKET_XY, POCKET_RIM+.011), True)
            gate.update(measurement(22, POCKET_XY+[.04, 0.], POCKET_RIM+.011), True)
            self.assertEqual(gate.carried, expected)

    def test_holes_are_not_filled_by_collision_hulls(self):
        nut = nut_spec()
        self.assertAlmostEqual(nut.mass, .00957)
        self.assertEqual(len(nut.pieces), 12)
        for part in nut.pieces:
            eq = ConvexHull(part.vertices).equations
            self.assertFalse(np.all(eq[:, :3]@[0., 0., NUT_HEIGHT/2]+eq[:, 3] <= 1e-10))
        coupon = fixtures()[0]
        self.assertTrue(coupon.box_primitives)
        self.assertEqual(len(coupon.pieces), 5)
        for part in coupon.pieces:
            eq = ConvexHull(part.vertices).equations
            self.assertFalse(np.all(eq[:, :3]@[0., 0., .020]+eq[:, 3] <= 1e-10))

    def test_reset_requires_whole_nut_below_rim_and_inside_rectangular_mouth(self):
        gate = RetrievalGate()
        gate.update(measurement(0, POCKET_XY+[.008, 0.], POCKET_FLOOR, False), False)
        self.assertFalse(gate.initial_in_pocket, "Center inside opening is insufficient when the nut penetrates a wall")
        gate = RetrievalGate()
        m = measurement(0, POCKET_XY, POCKET_FLOOR, False)
        m["payload"]["high"][2] = POCKET_RIM+.001
        gate.update(m, False)
        self.assertFalse(gate.initial_in_pocket)


if __name__ == "__main__":
    unittest.main()
