"""Load success needs actual jaw support in air, with the expected gravity load."""
import copy
import unittest
import numpy as np
from yam_arena import ik
from mechanism import load_spec, link_transforms, transform, jaw_angle_for_gap
from load_controller import TIP_DOWN
from load_gauge import MASS, BITE_HEIGHT
from load_score import LoadGate


class LoadScoreTests(unittest.TestCase):
    def measurement(self):
        spec = load_spec()
        links = link_transforms(spec, dict(jaw=jaw_angle_for_gap(spec, .010), fold_a=0., fold_b=0.))
        poses, points, forces = {}, [], []
        root = transform([.24, 0., .88], TIP_DOWN)
        for body, sign in (("jaw_a", 1), ("jaw_b", -1)):
            tr = root@links[body]
            poses[body] = np.r_[tr[:3, 3], ik.matrix_to_quat_xyzw(tr[:3, :3])].tolist()
            points.append((tr@[.047, sign*.0004, 0., 1.])[:3])
            forces.append(-sign*2.5*tr[:3, 1])
        origin = np.mean(points, axis=0)-[0., 0., BITE_HEIGHT]
        return {"link_poses": poses, "payload": {"name": "load_gauge",
                "pose": [*origin, 0., 0., 0., 1.], "mass_kg": MASS,
                "low": origin.tolist(), "velocity": [0.]*6,
                "contact_points_w": [p.tolist() for p in points]+[[None]*3 for _ in range(4)],
                "contact_forces_n": [f.tolist() for f in forces]+[[0.]*3 for _ in range(4)]}}

    def test_requires_two_seconds(self):
        gate = LoadGate()
        m = self.measurement()
        for _ in range(39):
            gate.update(m, True)
        self.assertFalse(gate.passed)
        gate.update(m, True)
        self.assertTrue(gate.passed)

    def test_rejects_support_shortcuts_and_wrong_load(self):
        for variant in ("no_deployment", "on_table", "robot_support", "wrong_mass", "shoulder", "moving"):
            gate, m = LoadGate(), copy.deepcopy(self.measurement())
            p = m["payload"]
            if variant == "on_table":
                p["low"][2] = .7475
            elif variant == "robot_support":
                p["contact_forces_n"][4][2] = 1.
            elif variant == "wrong_mass":
                p["mass_kg"] = .001
            elif variant == "shoulder":
                p["pose"][2] += .010
            elif variant == "moving":
                p["velocity"][2] = -.1
            for _ in range(50):
                gate.update(m, variant != "no_deployment")
            self.assertFalse(gate.passed, variant)

    def test_pose_window_handles_velocity_jitter_but_rejects_slow_drift(self):
        for drifting in (False, True):
            gate = LoadGate()
            for i in range(45):
                m = self.measurement()
                p = m["payload"]
                p["velocity"][3] = .35  # last-substep contact velocity
                dx = i*.0002 if drifting else 0.
                p["pose"][0] += dx
                p["low"][0] += dx
                for pose in m["link_poses"].values():
                    pose[0] += dx
                for point in p["contact_points_w"][:2]:
                    point[0] += dx
                gate.update(m, True)
            self.assertEqual(gate.passed, not drifting)


if __name__ == "__main__":
    unittest.main()
