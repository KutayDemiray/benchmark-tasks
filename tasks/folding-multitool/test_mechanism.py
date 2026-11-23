"""CPU unit checks; no simulator, GPU, hosted model, or environment mutation."""
import copy
import unittest
import numpy as np
from mechanism import load_spec, reset_joints, link_transforms, potential, effort, jaw_gap, jaw_angle_for_gap
from geometry import bounds, tool_links, mass_properties
from scorer import DeploymentScorer


class MechanicsTests(unittest.TestCase):
    def test_potential_derivative_and_dissipation(self):
        for j in load_spec()["joints"].values():
            lo, hi = j["limits_rad"]
            q = np.linspace(lo+1e-4, hi-1e-4, 501)
            eps = 1e-7
            numeric = (potential(q+eps, j)[0]-potential(q-eps, j)[0])/(2*eps)
            np.testing.assert_allclose(potential(q, j)[1], numeric, atol=1e-7)
            for velocity in (-2., -.03, 0., .03, 2.):
                energy_rate = (effort(q, velocity, j)+potential(q, j)[1])*velocity
                self.assertTrue(np.all(energy_rate <= 1e-12))

    def test_real_size_and_physical_inertia(self):
        s = load_spec()
        for q, length in ((reset_joints(s), .1016), (dict(jaw=0, fold_a=0, fold_b=0), .15875)):
            low, high = bounds(s, q)
            np.testing.assert_allclose(high-low, [length, .029972, .01651], atol=.0001)
        for body, pieces in tool_links(s).items():
            _, inertia = mass_properties(pieces, s["body_masses_kg"][body])
            eigenvalues = np.linalg.eigvalsh(inertia)
            self.assertGreater(eigenvalues.min(), 0.)
            self.assertLessEqual(eigenvalues.max(), eigenvalues.sum()/2+1e-12)

    def test_anchors_and_jaw_opening(self):
        s = load_spec()
        for angle in (0., -.3, -.7):
            poses = link_transforms(s, dict(jaw=angle, fold_a=.8, fold_b=-.7))
            g = s["geometry"]
            np.testing.assert_allclose(poses["handle_a"][:3, 3], [g["fold_pivot_x"], -g["fold_pivot_half_span"], 0])
            anchor = poses["jaw_b"]@np.array([g["fold_pivot_x"], g["fold_pivot_half_span"], 0, 1])
            np.testing.assert_allclose(poses["handle_b"][:3, 3], anchor[:3])
        for gap in (.001, .010, .014, .025):
            self.assertAlmostEqual(jaw_gap(s, jaw_angle_for_gap(s, gap)), gap, places=8)
        self.assertEqual(float(effort(-.4, 0., s["joints"]["jaw"])), 0.)


class ScorerTests(unittest.TestCase):
    def measurement(self, step=0, gap=.0007):
        return {"step": step, "q": dict(jaw=0., fold_a=0., fold_b=0.),
                "root_pos": [.4, 0, .9], "gap_m": gap, "tip_axis": [0., 0., -1.],
                "left_contact_names": ["link_6", "tip_left", "tip_right"],
                "right_contact_names": ["link_6", "tip_left", "tip_right"],
                "left_contact_forces_n": [[0, 0, 0], [0, 0, 2], [0, 0, -2]],
                "right_contact_forces_n": [[0, 0, 0], [0, 0, 2], [0, 0, -2]],
                "pads_n": [[0, 0, 2, 0], [0, 0, 2, 0], [0, 0, 0, 2], [0, 0, 0, 2]]}

    def feed(self, scorer, measurement, count):
        for _ in range(count):
            measurement["step"] += 1
            scorer.update(measurement)

    def test_needs_folded_history_and_two_handle_contacts(self):
        for variant in ("predeployed", "head_held", "single_pad", "collapsed", "forearm_support", "missing_contacts"):
            scorer = DeploymentScorer()
            m = self.measurement()
            if variant != "predeployed":
                m["q"].update(fold_a=np.pi, fold_b=-np.pi)
            scorer.update(m)
            m["q"].update(fold_a=0., fold_b=0.)
            if variant == "head_held":
                m["pads_n"][0][0] = 1.
            if variant == "single_pad":
                m["pads_n"][1][2] = 0.
            if variant == "collapsed":
                m["q"]["fold_a"] = .2
            if variant == "forearm_support":
                m["left_contact_forces_n"][0] = [0, 0, 1]
            if variant == "missing_contacts":
                del m["right_contact_names"]
            m["gap_m"] = .016
            self.feed(scorer, m, 20)
            m["gap_m"] = .0007
            self.feed(scorer, m, 40)
            self.assertEqual(scorer.score, 0., variant)

    def test_complete_cycle_is_partial_only(self):
        scorer = DeploymentScorer()
        m = self.measurement()
        m["q"].update(fold_a=np.pi, fold_b=-np.pi)
        scorer.update(m)
        m["q"].update(fold_a=0., fold_b=0.)
        m["gap_m"] = .016
        self.feed(scorer, m, 20)
        m["gap_m"] = .0007
        self.feed(scorer, m, 40)
        self.assertEqual(scorer.score, .25)
        self.assertTrue(scorer.snapshot()["deployment_cycle_pass"])
        self.assertFalse(scorer.snapshot()["loaded_gauge_pass"])
        self.assertFalse(scorer.snapshot()["retrieval_pass"])


if __name__ == "__main__":
    unittest.main()
