"""Reject mechanisms that move a payload without an opposing inner-face pinch."""
import unittest
import numpy as np
from yam_arena import ik
from mechanism import load_spec, link_transforms, jaw_angle_for_gap
from jaw_contact import inner_pinch


class JawContactTests(unittest.TestCase):
    def sample(self):
        spec = load_spec()
        links = link_transforms(spec, dict(jaw=jaw_angle_for_gap(spec, .010), fold_a=0, fold_b=0))
        poses, forces, points = {}, [], []
        for body, sign in (("jaw_a", 1), ("jaw_b", -1)):
            tr = links[body]
            poses[body] = np.r_[tr[:3, 3], ik.matrix_to_quat_xyzw(tr[:3, :3])].tolist()
            forces.append(-sign*2*tr[:3, 1])
            points.append((tr@np.array([.047, sign*.0004, 0., 1.]))[:3])
        return spec, poses, np.array(forces), np.array(points)

    def test_opposed_inner_faces(self):
        self.assertTrue(inner_pinch(*self.sample()))

    def test_single_jaw_or_carried_against_tool(self):
        for variant in ("single", "same_direction", "outer", "base", "missing_point"):
            spec, poses, f, p = self.sample()
            if variant == "single":
                f[1] = 0.
            elif variant == "same_direction":
                f[1] = f[0]
            elif variant == "outer":
                p[0][1] = .005
            elif variant == "base":
                p[0][0] = .015
            elif variant == "missing_point":
                p[0] = np.nan
            self.assertFalse(inner_pinch(spec, poses, f, p), variant)


if __name__ == "__main__":
    unittest.main()
