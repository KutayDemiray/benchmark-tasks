"""Integrated outcome regressions, using synthetic measurements, not rollouts."""
import copy
import unittest
import numpy as np
from mechanism import load_spec, jaw_angle_for_gap
from scorer import DeploymentScorer
from test_retrieval import measurement
from retrieval_scene import POCKET_XY, POCKET_FLOOR, POCKET_RIM, TRAY_XY, TRAY_FLOOR


def observed(step, xy, bottom, pinch=True):
    m = measurement(step, xy, bottom, pinch)
    m.update(q=dict(jaw=jaw_angle_for_gap(load_spec(), .010), fold_a=0., fold_b=0.),
             root_pos=m["link_poses"]["jaw_a"][:3], gap_m=.010,
             tip_axis=[0., 0., -1.],
             pads_n=[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
    for side in ("left", "right"):
        m[side+"_contact_names"] = ["link_6", "tip_left", "tip_right"]
        m[side+"_contact_forces_n"] = [[0, 0, 0], [1, 0, 0], [-1, 0, 0]]
    if step == 0:
        m["q"].update(jaw=0., fold_a=np.pi, fold_b=-np.pi)
        m["gap_m"] = 0.
    return m


def retrieval_sequence():
    yield observed(0, POCKET_XY, POCKET_FLOOR, False)
    yield observed(1, POCKET_XY, POCKET_FLOOR)
    for step in range(2, 22):
        yield observed(step, POCKET_XY, POCKET_RIM+.011)
    yield observed(22, POCKET_XY+[.04, 0.], POCKET_RIM+.011)
    yield observed(23, TRAY_XY, TRAY_FLOOR+.01)
    for step in range(24, 64):
        yield observed(step, TRAY_XY, TRAY_FLOOR, False)


class PrimaryScoreTests(unittest.TestCase):
    def evaluate(self, change=lambda m: None):
        scorer = DeploymentScorer()
        scores = []
        for m in retrieval_sequence():
            change(m)
            scorer.update(m)
            scores.append(scorer.score)
        return scorer, scores

    def test_full_retrieval_without_empty_cycle_or_orientation_dwell(self):
        scorer, scores = self.evaluate()
        self.assertEqual(scores[21], 0.)  # Extraction alone is not carry.
        self.assertEqual(scores[22], .5)
        self.assertEqual(scores[62], .5)  # Still needs all 40 release frames.
        self.assertEqual(scores[63], 1.)
        self.assertFalse(scorer.open_seen)
        self.assertFalse(scorer.cycle_seen)
        self.assertFalse(scorer.tip_down_seen)

    def test_retrieval_does_not_prescribe_arm_assignment(self):
        assignments = (
            [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]],
            [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        for pads in assignments:
            with self.subTest(pads=pads):
                scorer, _ = self.evaluate(lambda m: m.update(pads_n=copy.deepcopy(pads)))
                self.assertEqual(scorer.score, 1.)
                self.assertFalse(scorer.cycle_seen)

    def test_invalid_tool_use_cannot_extract_then_score_by_placement(self):
        def mutate(m, variant):
            if variant == "predeployed" and m["step"] == 0:
                m["q"].update(fold_a=0., fold_b=0.)
            if m["step"] == 0:
                return
            if variant == "fold_collapse":
                m["q"]["fold_a"] = .2
            elif variant == "one_handle":
                for p in m["pads_n"]:
                    p[3] = 0.
            elif variant == "one_finger":
                m["pads_n"] = [[0, 0, 1, 1]]+[[0, 0, 0, 0] for _ in range(3)]
            elif variant == "head_held":
                m["pads_n"][0][0] = 1.
            elif variant == "forearm_support":
                m["left_contact_forces_n"][0] = [0, 0, 1]
            elif variant == "missing_contacts":
                del m["right_contact_names"]
            elif variant == "single_jaw":
                m["payload"]["contact_forces_n"][1] = [0., 0., 0.]
            elif variant == "robot_support":
                m["payload"]["contact_forces_n"][4] = [0., 0., .094]
            elif variant == "fixture_support":
                m["payload"]["unattributed_normal_n"] = [0., 0., .094]
            elif variant == "direct_placement":
                m["payload"]["contact_forces_n"] = [[0., 0., 0.] for _ in range(6)]
        for variant in ("predeployed", "fold_collapse", "one_handle", "one_finger",
                        "head_held", "forearm_support", "missing_contacts", "single_jaw",
                        "robot_support", "fixture_support", "direct_placement"):
            with self.subTest(variant=variant):
                scorer, _ = self.evaluate(lambda m: mutate(m, variant))
                self.assertEqual(scorer.score, 0.)
                self.assertFalse(scorer.retrieval.carried)

    def test_final_clearance_and_stability_remain_required(self):
        for variant in ("clearance", "motion", "drop_elsewhere"):
            def mutate(m):
                if variant == "drop_elsewhere" and m["step"] == 23:
                    m.update(observed(23, POCKET_XY+[.06, 0.], POCKET_FLOOR, False))
                if m["step"] >= 24:
                    if variant == "clearance":
                        m["payload"]["tool_robot_clearance_m"] = .019
                    elif variant == "motion":
                        m["payload"]["velocity"][0] = .006
            with self.subTest(variant=variant):
                scorer, _ = self.evaluate(mutate)
                self.assertEqual(scorer.score, .5)
                self.assertFalse(scorer.retrieval.passed)


if __name__ == "__main__":
    unittest.main()
